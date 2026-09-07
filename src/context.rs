// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

//! [Context] and [Ptr] together provide memory management for `pliron`.

use core::{
    any::Any,
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::Display,
    hash::Hash,
    marker::PhantomData,
    num::NonZeroU64,
};

use crate::{
    arg_error_noloc,
    basic_block::BasicBlock,
    common_traits::Verify,
    dialect::{Dialect, DialectName},
    identifier::Identifier,
    operation::Operation,
    printable::{self, Printable},
    region::Region,
    result::Result,
    std_deps::sync::{LazyLock, Mutex},
    storage_uniquer::UniqueStore,
    r#type::TypeObj,
    uniqued_any::UniquedAny,
    utils::table::{HMap, HSet, IMap},
    verify_err_noloc,
};
use alloc::{boxed::Box, format, string::ToString, vec, vec::Vec};
use slotmap::{SlotMap, new_key_type};

new_key_type! {
    /// The index type for the [SlotMap] used to store IR objects.
    pub struct ArenaIndex;
}

new_key_type! {
    /// The index type for the [SlotMap] used to store auxiliary data.
    pub struct AuxDataIndex;
}

impl Display for ArenaIndex {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Opaque observation of one [Context]'s monotonic IR mutation-attempt
/// counter. Callers can compare observations but cannot manufacture them.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IrMutationAttemptEpoch(u64);

impl IrMutationAttemptEpoch {
    /// Numeric value for diagnostics and monotonicity checks.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// The mutation-attempt counter exhausted its bounded u64 representation.
/// Once exhausted, the context can no longer certify a mutation-free interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("IR mutation-attempt epoch exhausted")]
pub struct IrMutationAttemptEpochExhausted;

/// The process-local [Context] identity sequence was exhausted.
///
/// Context identities are never reused. Once all non-zero `u64` identities
/// have been issued, constructing another context fails permanently.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("context identity space exhausted")]
pub struct ContextIdentityExhausted;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ContextIdentity(NonZeroU64);

static NEXT_CONTEXT_IDENTITY: LazyLock<Mutex<Option<NonZeroU64>>> =
    LazyLock::new(|| Mutex::new(NonZeroU64::new(1)));

fn take_context_identity(
    next: &mut Option<NonZeroU64>,
) -> core::result::Result<ContextIdentity, ContextIdentityExhausted> {
    let identity = next.take().ok_or(ContextIdentityExhausted)?;
    *next = identity.get().checked_add(1).and_then(NonZeroU64::new);
    Ok(ContextIdentity(identity))
}

fn allocate_context_identity() -> core::result::Result<ContextIdentity, ContextIdentityExhausted> {
    #[cfg(feature = "std")]
    let mut next = NEXT_CONTEXT_IDENTITY
        .lock()
        .map_err(|_| ContextIdentityExhausted)?;
    #[cfg(not(feature = "std"))]
    let mut next = NEXT_CONTEXT_IDENTITY.lock();
    take_context_identity(&mut next)
}

/// An arena allocation pool for IR objects.
pub type Arena<T> = SlotMap<ArenaIndex, RefCell<T>>;

/// A context stores all IR data of this compilation session.
pub struct Context {
    /// Unforgeable process-local identity carried by every arena pointer.
    identity: ContextIdentity,
    /// Monotonic counter incremented before every mutation route that can be
    /// reached through a shared [Context] reference, and before arena
    /// allocation/deallocation through an exclusive reference.
    ir_mutation_attempt_epoch: Cell<u64>,
    /// Once the counter is exhausted, observations fail closed permanently.
    ir_mutation_attempt_epoch_exhausted: Cell<bool>,
    /// A unique number for each `Value` in the context.
    /// Serves as a generation counter against accessing invalid `Value`s.
    pub(crate) value_counter: Cell<u64>,
    /// A unique number for each `Use` in the context.
    /// Serves as a generation counter against accessing invalid `Use`s.
    pub(crate) use_counter: Cell<u64>,
    /// Allocation pool for [Operation]s.
    pub(crate) operations: Arena<Operation>,
    /// Allocation pool for [BasicBlock]s.
    pub(crate) basic_blocks: Arena<BasicBlock>,
    /// Allocation pool for [Region]s.
    pub(crate) regions: Arena<Region>,
    /// Registered [Dialect]s.
    pub(crate) dialects: HMap<DialectName, Dialect>,
    /// Storage for uniqued [TypeObj]s.
    pub(crate) type_store: UniqueStore<TypeObj>,
    /// Storage for other uniqued objects.
    pub(crate) uniqued_any_store: UniqueStore<UniquedAny>,
    /// Arbitrary data storage. Use [Self::aux_data_map] for dictionary access.
    pub aux_data: SlotMap<AuxDataIndex, Box<dyn Any>>,
    /// A dictionary with keys mapping to an index in [Self::aux_data].
    pub aux_data_map: HMap<Identifier, AuxDataIndex>,
}

impl Context {
    pub fn new() -> Context {
        Self::default()
    }

    /// Construct a context, returning a typed error if its process-local
    /// identity cannot be allocated without reuse.
    pub fn try_new() -> core::result::Result<Context, ContextIdentityExhausted> {
        let identity = allocate_context_identity()?;
        Ok(Self::with_identity(identity))
    }

    pub(crate) fn identity(&self) -> ContextIdentity {
        self.identity
    }

    /// Observe the current IR mutation-attempt epoch.
    ///
    /// The epoch advances *before* PLIRON returns mutable access or mutates a
    /// context-owned IR store. It therefore records failed mutable borrows and
    /// mutate-then-restore sequences as well as retained mutations. Equality
    /// only has meaning for two observations of the same [Context].
    pub fn ir_mutation_attempt_epoch(
        &self,
    ) -> core::result::Result<IrMutationAttemptEpoch, IrMutationAttemptEpochExhausted> {
        if self.ir_mutation_attempt_epoch_exhausted.get() {
            Err(IrMutationAttemptEpochExhausted)
        } else {
            Ok(IrMutationAttemptEpoch(self.ir_mutation_attempt_epoch.get()))
        }
    }

    /// Records an attempted mutation before any mutable access is granted.
    pub(crate) fn note_ir_mutation_attempt(&self) {
        if self.ir_mutation_attempt_epoch_exhausted.get() {
            return;
        }
        match self.ir_mutation_attempt_epoch.get().checked_add(1) {
            Some(next) => self.ir_mutation_attempt_epoch.set(next),
            None => self.ir_mutation_attempt_epoch_exhausted.set(true),
        }
    }

    /// Is the IR in this context empty?
    /// An IR is considered empty if it has no operations, basic blocks, or regions.
    /// This does not check for types, dialects, ops, or aux_data stored in the context.
    pub fn is_ir_empty(&self) -> bool {
        self.operations.is_empty() && self.basic_blocks.is_empty() && self.regions.is_empty()
    }

    /// Get a unique number for a new value.
    pub(crate) fn get_new_value_uid(&self) -> u64 {
        self.note_ir_mutation_attempt();
        let uid = self.value_counter.get();
        self.value_counter.set(uid + 1);
        uid
    }

    /// Get a unique number for a new use.
    pub(crate) fn get_new_use_uid(&self) -> u64 {
        self.note_ir_mutation_attempt();
        let uid = self.use_counter.get();
        self.use_counter.set(uid + 1);
        uid
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::try_new().expect("context identity space exhausted")
    }
}

impl Context {
    fn with_identity(identity: ContextIdentity) -> Self {
        let mut ctx = Context {
            identity,
            ir_mutation_attempt_epoch: Cell::new(0),
            ir_mutation_attempt_epoch_exhausted: Cell::new(false),
            value_counter: Cell::new(0),
            use_counter: Cell::new(0),
            operations: Arena::default(),
            basic_blocks: Arena::default(),
            regions: Arena::default(),
            dialects: HMap::default(),
            type_store: UniqueStore::default(),
            uniqued_any_store: UniqueStore::default(),
            aux_data: SlotMap::with_key(),
            aux_data_map: HMap::default(),
        };

        // Verify that all dictionary keys are unique.
        if let Err(err) = &*DICT_KEYS_VERIFIER {
            panic!("{}", err.err);
        }

        // Run all context registrations
        for registration in get_context_registrations() {
            registration(&mut ctx);
        }

        ctx
    }
}

pub(crate) mod private {
    use super::*;

    /// An IR object owned by Context
    pub trait ArenaObj
    where
        Self: Sized,
    {
        /// Get the arena that has allocated this object.
        fn get_arena(ctx: &Context) -> &Arena<Self>;
        /// Get the arena that has allocated this object.
        fn get_arena_mut(ctx: &mut Context) -> &mut Arena<Self>;
        /// Get a Ptr to self.
        fn get_self_ptr(&self, ctx: &Context) -> Ptr<Self>;
        /// If this object contains any ArenaObj itself, it must dealloc()
        /// all of those sub-objects. This is called when self is deallocated.
        fn dealloc_sub_objects(ptr: Ptr<Self>, ctx: &mut Context);

        /// Allocates object on the arena, given a creator function.
        fn alloc<T: FnOnce(Ptr<Self>) -> Self>(ctx: &mut Context, f: T) -> Ptr<Self> {
            ctx.note_ir_mutation_attempt();
            let owner = ctx.identity;
            let creator = move |idx: ArenaIndex| {
                let t = f(Ptr::<Self> {
                    owner,
                    idx,
                    _dummy: PhantomData::<Self>,
                });
                RefCell::new(t)
            };
            Ptr::<Self> {
                owner,
                idx: Self::get_arena_mut(ctx).insert_with_key(creator),
                _dummy: PhantomData,
            }
        }

        /// Deallocates this object from the arena.
        fn dealloc(ptr: Ptr<Self>, ctx: &mut Context) {
            ptr.assert_owner(ctx);
            ctx.note_ir_mutation_attempt();
            Self::dealloc_sub_objects(ptr, ctx);
            Self::get_arena_mut(ctx).remove(ptr.idx);
        }
    }
}

use private::ArenaObj;

/// Pointer to an IR Object owned by Context.
pub struct Ptr<T: ArenaObj> {
    owner: ContextIdentity,
    pub(crate) idx: ArenaIndex,
    pub(crate) _dummy: PhantomData<T>,
}

impl<T: ArenaObj> core::fmt::Debug for Ptr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ptr<{}>[{}]", core::any::type_name::<T>(), self.idx)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("Attempt to dereference a dangling Ptr")]
pub struct DanglingPtrDerefError;

/// A [Ptr] was paired with a [Context] that does not own it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("attempt to dereference a Ptr with a non-owning Context")]
pub struct ForeignContextPtrDerefError;

impl<'a, T: ArenaObj> Ptr<T> {
    fn is_owned_by(&self, ctx: &Context) -> bool {
        self.owner == ctx.identity
    }

    #[track_caller]
    fn assert_owner(&self, ctx: &Context) {
        assert!(
            self.is_owned_by(ctx),
            "Attempt to use a Ptr with a non-owning Context"
        );
    }

    fn require_owner(&self, ctx: &Context) -> Result<()> {
        if self.is_owned_by(ctx) {
            Ok(())
        } else {
            Err(arg_error_noloc!(ForeignContextPtrDerefError))
        }
    }

    /// Borrow the inner [RefCell] and return a [Ref] to the pointee.
    /// The borrow is live as long as the returned [Ref] lives.
    /// Panics on dangling [Ptr] or borrow interior mutability errors.
    /// Run `cargo` with options `+nightly -Zbuild-std -Zbuild-std-features="debug_refcell"`
    /// to enable printing the interior mutability violation locations.
    #[track_caller]
    pub fn deref(&self, ctx: &'a Context) -> Ref<'a, T> {
        self.assert_owner(ctx);
        T::get_arena(ctx)
            .get(self.idx)
            .expect("Dangling Ptr deref")
            .borrow()
    }

    /// Mutably borrow the inner [RefCell] and return a [RefMut] to the pointee.
    /// The borrow is live as long as the returned [RefMut] lives.
    /// Panics on dangling [Ptr] or borrow interior mutability errors.
    /// Run `cargo` with options `+nightly -Zbuild-std -Zbuild-std-features="debug_refcell"`
    /// to enable printing the interior mutability violation locations.
    #[track_caller]
    pub fn deref_mut(&self, ctx: &'a Context) -> RefMut<'a, T> {
        self.assert_owner(ctx);
        ctx.note_ir_mutation_attempt();
        T::get_arena(ctx)
            .get(self.idx)
            .expect("Dangling Ptr deref_mut")
            .borrow_mut()
    }

    /// Try and borrow the inner [RefCell] and return a [Ref] to the pointee.
    /// The borrow is live as long as the returned [Ref] lives.
    /// If [Ptr] is dangling or already mutably borrowed, an [Error](crate::result::Error)
    /// with [DanglingPtrDerefError] or [BorrowError](core::cell::BorrowError) is returned.
    pub fn try_deref(&self, ctx: &'a Context) -> Result<Ref<'a, T>> {
        self.require_owner(ctx)?;
        T::get_arena(ctx)
            .get(self.idx)
            .ok_or_else(|| arg_error_noloc!(DanglingPtrDerefError))?
            .try_borrow()
            .map_err(|err| arg_error_noloc!(err))
    }

    /// Try and mutably borrow the inner [RefCell] and return a [RefMut] to the pointee.
    /// The borrow is live as long as the returned [RefMut] lives.
    /// If [Ptr] is dangling or already borrowed, an [Error](crate::result::Error)
    /// with [DanglingPtrDerefError] or [BorrowMutError](core::cell::BorrowMutError) is returned.
    pub fn try_deref_mut(&self, ctx: &'a Context) -> Result<RefMut<'a, T>> {
        self.require_owner(ctx)?;
        ctx.note_ir_mutation_attempt();
        T::get_arena(ctx)
            .get(self.idx)
            .ok_or_else(|| arg_error_noloc!(DanglingPtrDerefError))?
            .try_borrow_mut()
            .map_err(|err| arg_error_noloc!(err))
    }

    /// Create a unique (to the arena) name based on the arena index.
    pub(crate) fn make_name(&self, name_base: &str) -> Identifier {
        let idx = format!("{}", self.idx);
        (name_base.to_string() + &idx).try_into().unwrap()
    }
}

impl<T: ArenaObj> Clone for Ptr<T> {
    fn clone(&self) -> Ptr<T> {
        *self
    }
}

impl<T: ArenaObj> Copy for Ptr<T> {}

impl<T: ArenaObj> PartialEq for Ptr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.owner == other.owner && self.idx == other.idx
    }
}

impl<T: ArenaObj> Eq for Ptr<T> {}

impl<T: ArenaObj + 'static> Hash for Ptr<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.owner.hash(state);
        self.idx.hash(state);
    }
}

impl<T: ArenaObj + Printable> Printable for Ptr<T> {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.deref(ctx).fmt(ctx, state, f)
    }
}

impl<T: ArenaObj + Verify> Verify for Ptr<T> {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.try_deref(ctx)?.verify(ctx)
    }
}

#[doc(hidden)]
/// Declaration of a static [Identifier] for use as a dictionary key.
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct DictKeyId {
    /// The [Identifier] itself.
    pub id: Identifier,
    /// The file where this key was declared.
    pub file: &'static str,
    /// The line where this key was declared.
    pub line: u32,
    /// The column where this key was declared.
    pub column: u32,
}

/// These represent registrations that happen automatically at link time.
/// Every dialect, op, type, and attribute that are linked into your code
/// will register themselves using this type.
pub type ContextRegistration = fn(&mut Context);

#[doc(hidden)]
/// `pliron` uses dictionaries indexed by static [Identifier]s in many places,
/// such as [Context::aux_data_map], and the states of [Printable](crate::printable::State)
/// and [Parsable](crate::parsable::State). To avoid collisions in these [Identifier]s,
/// we use the [crate::dict_key!] macro to verify that all such keys declared using the macro
/// are unique. The macro adds the keys to this static slice, which is then verified
/// when a [Context] is created.
#[cfg(not(target_family = "wasm"))]
pub mod statics {
    use super::*;

    #[::pliron::linkme::distributed_slice]
    #[linkme(crate = ::pliron::linkme)]
    pub static DICT_KEY_IDS: [LazyLock<DictKeyId>];

    pub fn get_dict_key_ids() -> impl Iterator<Item = &'static LazyLock<DictKeyId>> {
        DICT_KEY_IDS.iter()
    }

    #[::pliron::linkme::distributed_slice]
    #[linkme(crate = ::pliron::linkme)]
    pub static CONTEXT_REGISTRATIONS: [ContextRegistration];

    pub fn get_context_registrations() -> impl Iterator<Item = &'static ContextRegistration> {
        CONTEXT_REGISTRATIONS.iter()
    }
}

#[cfg(target_family = "wasm")]
pub mod statics {
    use super::*;
    use crate::InventoryWrapper;

    ::pliron::inventory::collect!(InventoryWrapper<LazyLock<DictKeyId>>);

    pub fn get_dict_key_ids() -> impl Iterator<Item = &'static LazyLock<DictKeyId>> {
        ::pliron::inventory::iter::<InventoryWrapper<LazyLock<DictKeyId>>>().map(|llw| llw.0)
    }

    ::pliron::inventory::collect!(InventoryWrapper<ContextRegistration>);

    pub fn get_context_registrations() -> impl Iterator<Item = &'static ContextRegistration> {
        ::pliron::inventory::iter::<InventoryWrapper<ContextRegistration>>().map(|llw| llw.0)
    }
}

pub use statics::*;

#[doc(hidden)]
pub static DICT_KEYS_VERIFIER: LazyLock<Result<()>> = LazyLock::new(verify_dict_keys);

#[doc(hidden)]
/// Collect `(owner, __all_verifiers)` entries into an ordered verifier map.
///
/// Each owner (op/type/attribute) can contribute verifiers through multiple interfaces.
/// This helper preserves interface dependency order (as returned by each `__all_verifiers`
/// function) while deduplicating verifier function pointers.
pub(crate) fn collect_deduped_interface_verifiers<Id, AllVerifiers, Verifier>(
    interface_verifiers: impl Iterator<Item = &'static (Id, AllVerifiers)>,
) -> HMap<Id, Vec<Verifier>>
where
    Id: Eq + Hash + Clone + 'static,
    AllVerifiers: Fn() -> Vec<Verifier> + Clone + 'static,
    Verifier: Eq + Hash + Clone,
{
    let mut grouped = IMap::default();
    for entry in interface_verifiers {
        let (id, all_verifiers_for_interface) = entry;
        grouped
            .entry(id.clone())
            .and_modify(|verifiers: &mut Vec<AllVerifiers>| {
                verifiers.push(all_verifiers_for_interface.clone())
            })
            .or_insert(vec![all_verifiers_for_interface.clone()]);
    }

    // Remove duplicates (best effort as rustc may inline functions, resulting in different pointers).
    // Relies on `__all_verifiers` returning the super-verifiers followed by self verifier
    // to ensure that super-interfaces are verified first.
    grouped
        .into_iter()
        .map(|(id, verifiers)| {
            let mut dedupd_verifiers = Vec::new();
            let mut seen = HSet::default();
            for verifier_fn_list in verifiers {
                for verifier in verifier_fn_list() {
                    if seen.insert(verifier.clone()) {
                        dedupd_verifiers.push(verifier);
                    }
                }
            }
            (id, dedupd_verifiers)
        })
        .collect()
}

#[doc(hidden)]
/// Verify that all dictionary keys are unique. This is called when a [Context] is created.
/// If any duplicate keys are found, a panic is raised with the file, line, and column
/// information of the duplicate keys.
pub fn verify_dict_keys() -> Result<()> {
    let mut seen: HMap<Identifier, (&'static str, u32, u32)> = HMap::default();
    for key in get_dict_key_ids() {
        if let Some((file, line, column)) = seen.get(&key.id) {
            return verify_err_noloc!(
                "Duplicate dictionary key \"{}\" declared in {}:{}:{} and {}:{}:{}",
                key.id,
                file,
                line,
                column,
                key.file,
                key.line,
                key.column
            );
        }
        seen.insert(key.id.clone(), (key.file, key.line, key.column));
    }
    Ok(())
}

/// A macro to declare a static [Identifier] for use as a dictionary key.
///
/// Usage:
/// ```
/// # use pliron::dict_key;
/// dict_key!(MY_KEY, "my_key");
/// let mut ctx = pliron::context::Context::new();
/// let aux_data_index = ctx.aux_data.insert(Box::new(42));
/// ctx.aux_data_map.insert(MY_KEY.clone(), aux_data_index);
/// assert_eq!(ctx.aux_data[aux_data_index].downcast_ref::<i32>(), Some(&42));
/// assert_eq!(ctx.aux_data_map[&*MY_KEY], aux_data_index);
/// ```
/// Here, `MY_KEY` is the name of the static variable, and `"my_key"` is the
/// string value of the [Identifier]. The macro will create a static variable
/// of type [`LazyLock<Identifier>`](LazyLock) with the name `MY_KEY`.
#[macro_export]
macro_rules! dict_key {
    (   $(#[$outer:meta])*
        $decl:ident, $name:expr
    ) => {
        // Create a static variable linked to the DICT_KEY_IDS slice
        // to ensure that all keys are unique.
        // The static variable is created in a separate anonymous module.
        const _: () = {
            #[cfg_attr(not(target_family = "wasm"),
                ::pliron::linkme::distributed_slice(::pliron::context::DICT_KEY_IDS), linkme(crate = ::pliron::linkme))]
            pub static $decl: $crate::std_deps::sync::LazyLock<::pliron::context::DictKeyId> =
                $crate::std_deps::sync::LazyLock::new(|| ::pliron::context::DictKeyId {
                    id: $name.try_into().unwrap(),
                    file: file!(),
                    line: line!(),
                    column: column!(),
                });

            #[cfg(target_family = "wasm")]
            ::pliron::inventory::submit! {
                ::pliron::InventoryWrapper(&$decl)
            }
        };
        $(#[$outer])*
        // Create a static variable with the provided name to access the identifier.
        pub static $decl: $crate::std_deps::sync::LazyLock<::pliron::identifier::Identifier> =
            $crate::std_deps::sync::LazyLock::new(|| $name.try_into().unwrap());
    };
}

/// A macro to register a [ContextRegistration]. The argument function
/// will be called with a mutable reference to the [Context] when a [Context] is created.
/// Use this outside of any function (e.g. in the module scope).
///
/// Usage:
/// ```
/// use pliron::context_registration;
/// context_registration!(my_registration_fn);
/// fn my_registration_fn(_: &mut pliron::context::Context) {}
/// ```
/// Here, `my_registration_fn` is a function matching [ContextRegistration].
#[macro_export]
macro_rules! context_registration {
    (   $(#[$outer:meta])*
        $registration:expr
    ) => {
        const _: () = {
            $(#[$outer])*
            #[cfg_attr(not(target_family = "wasm"),
                ::pliron::linkme::distributed_slice(::pliron::context::CONTEXT_REGISTRATIONS), linkme(crate = ::pliron::linkme))]
            static CONTEXT_REGISTRATION: ::pliron::context::ContextRegistration = $registration;

            #[cfg(target_family = "wasm")]
            ::pliron::inventory::submit! {
                ::pliron::InventoryWrapper(&CONTEXT_REGISTRATION)
            }
        };
    };
}

#[cfg(test)]
mod mutation_epoch_tests {
    use super::{
        Context, ContextIdentityExhausted, DanglingPtrDerefError, ForeignContextPtrDerefError, Ptr,
        private::ArenaObj, take_context_identity,
    };
    use crate::{
        attribute::AttrObj,
        basic_block::BasicBlock,
        builtin::{
            attributes::{DictAttr, TypeAttr, VecAttr},
            ops::{ConstantOp, FuncOp, ModuleOp},
            types::{FunctionType, IntegerType, Signedness, UnitType},
        },
        common_traits::Verify,
        graph::dominance::DomInfo,
        irbuild::dialect_conversion::{
            DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
        },
        irbuild::{
            inserter::{BlockInsertionPoint, IRInserter, Inserter, OpInsertionPoint},
            listener::Recorder,
            rewriter::{IRRewriter, Rewriter},
        },
        linked_list::{ContainsLinkedList, LinkedList, append_linked_list},
        op::Op,
        operation::{Operation, verify_operation},
        opts::{dce::dce, mem2reg::mem2reg, simplify_cfg::simplify_cfg},
        pass::AnalysisManager,
        region::Region,
        r#type::{ForeignContextTypeHandleError, TypeHandle, Typed, TypedHandle},
        uniqued_any,
        value::{DefiningEntity, Use, Value},
    };
    use alloc::{boxed::Box, vec, vec::Vec};
    use core::{marker::PhantomData, num::NonZeroU64, panic::AssertUnwindSafe};

    struct ArenaFixture {
        operation: Ptr<Operation>,
        region: Ptr<Region>,
        block: Ptr<BasicBlock>,
        value: Value,
    }

    #[derive(Eq, PartialEq)]
    struct ArenaGraphSnapshot {
        operation_count: usize,
        region_count: usize,
        block_count: usize,
        value_counter: u64,
        use_counter: u64,
        operands: Vec<Value>,
        successors: Vec<Ptr<BasicBlock>>,
        regions: Vec<Ptr<Region>>,
        operation_next: Option<Ptr<Operation>>,
        operation_previous: Option<Ptr<Operation>>,
        operation_container: Option<Ptr<BasicBlock>>,
        region_parent: Ptr<Operation>,
        region_head: Option<Ptr<BasicBlock>>,
        region_tail: Option<Ptr<BasicBlock>>,
        block_next: Option<Ptr<BasicBlock>>,
        block_previous: Option<Ptr<BasicBlock>>,
        block_container: Option<Ptr<Region>>,
        block_uses: Vec<Use<Ptr<BasicBlock>>>,
        value_uses: Vec<Use<Value>>,
    }

    fn arena_fixture(context: &mut Context, name: &str) -> ArenaFixture {
        let module = ModuleOp::new(context, name.try_into().unwrap());
        let operation = module.get_operation();
        let region = operation.deref(context).get_region(0);
        let block = region.deref(context).get_head().unwrap();
        let ty: TypeHandle = IntegerType::get(context, 32, Signedness::Signless).into();
        let value_block = BasicBlock::new(context, None, vec![ty]);
        let value = value_block.deref(context).get_argument(0);
        ArenaFixture {
            operation,
            region,
            block,
            value,
        }
    }

    fn is_foreign_context_error(error: &crate::result::Error) -> bool {
        error
            .err
            .downcast_ref::<ForeignContextPtrDerefError>()
            .is_some()
    }

    fn is_dangling_pointer_error(error: &crate::result::Error) -> bool {
        error.err.downcast_ref::<DanglingPtrDerefError>().is_some()
    }

    fn is_foreign_type_error(error: &crate::result::Error) -> bool {
        error
            .err
            .downcast_ref::<ForeignContextTypeHandleError>()
            .is_some()
    }

    fn expect_error<T>(result: crate::result::Result<T>) -> crate::result::Error {
        match result {
            Ok(_) => panic!("expected operation to fail"),
            Err(error) => error,
        }
    }

    fn epoch(ctx: &Context) -> u64 {
        ctx.ir_mutation_attempt_epoch().unwrap().value()
    }

    macro_rules! assert_foreign_mutation_rejected {
        ($owner:ident, $owner_fixture:ident, $foreign:ident, $foreign_fixture:ident, $action:block) => {{
            let owner_epoch = epoch(&$owner);
            let foreign_epoch = epoch(&$foreign);
            let owner_graph = graph_snapshot(&$owner, &$owner_fixture);
            let foreign_graph = graph_snapshot(&$foreign, &$foreign_fixture);
            assert!(
                std::panic::catch_unwind(AssertUnwindSafe(|| $action)).is_err(),
                "foreign composite mutation unexpectedly succeeded"
            );
            assert_eq!(epoch(&$owner), owner_epoch);
            assert_eq!(epoch(&$foreign), foreign_epoch);
            assert!(graph_snapshot(&$owner, &$owner_fixture) == owner_graph);
            assert!(graph_snapshot(&$foreign, &$foreign_fixture) == foreign_graph);
        }};
    }

    macro_rules! assert_foreign_result_rejected {
        ($owner:ident, $owner_fixture:ident, $foreign:ident, $foreign_fixture:ident, $action:expr) => {{
            let owner_epoch = epoch(&$owner);
            let foreign_epoch = epoch(&$foreign);
            let owner_graph = graph_snapshot(&$owner, &$owner_fixture);
            let foreign_graph = graph_snapshot(&$foreign, &$foreign_fixture);
            let error = expect_error($action);
            assert!(is_foreign_context_error(&error));
            assert_eq!(epoch(&$owner), owner_epoch);
            assert_eq!(epoch(&$foreign), foreign_epoch);
            assert!(graph_snapshot(&$owner, &$owner_fixture) == owner_graph);
            assert!(graph_snapshot(&$foreign, &$foreign_fixture) == foreign_graph);
        }};
    }

    #[test]
    fn equal_slot_type_handles_are_context_owned_and_fail_fallibly() {
        let owner = Context::try_new().unwrap();
        let foreign = Context::try_new().unwrap();
        let owner_type: TypeHandle = IntegerType::get(&owner, 32, Signedness::Signless).into();
        let foreign_type: TypeHandle = UnitType::get(&foreign).into();

        assert_eq!(owner_type.storage_index(), foreign_type.storage_index());
        assert_ne!(owner_type, foreign_type);
        assert!(owner_type.try_deref(&owner).unwrap().is::<IntegerType>());
        assert!(foreign_type.try_deref(&foreign).unwrap().is::<UnitType>());

        let owner_epoch = epoch(&owner);
        let foreign_epoch = epoch(&foreign);
        for error in [
            expect_error(foreign_type.try_deref(&owner)),
            expect_error(foreign_type.try_deref_mut(&owner)),
            expect_error(foreign_type.verify(&owner)),
            expect_error(TypedHandle::<UnitType>::from_handle(foreign_type, &owner)),
        ] {
            assert!(is_foreign_type_error(&error));
        }
        assert_eq!(epoch(&owner), owner_epoch);
        assert_eq!(epoch(&foreign), foreign_epoch);

        let same_owner = Context::try_new().unwrap();
        let same_foreign = Context::try_new().unwrap();
        let same_owner_type: TypeHandle =
            IntegerType::get(&same_owner, 32, Signedness::Signless).into();
        let same_foreign_type: TypeHandle =
            IntegerType::get(&same_foreign, 32, Signedness::Signless).into();
        assert_eq!(
            same_owner_type.storage_index(),
            same_foreign_type.storage_index()
        );
        assert_ne!(same_owner_type, same_foreign_type);
        let error = expect_error(same_foreign_type.try_preflight(&same_owner));
        assert!(is_foreign_type_error(&error));
    }

    #[test]
    fn equal_slot_foreign_types_are_rejected_before_core_ir_mutation() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_type: TypeHandle = IntegerType::get(&owner, 32, Signedness::Signless).into();
        let foreign_type: TypeHandle = UnitType::get(&foreign).into();
        assert_eq!(owner_type.storage_index(), foreign_type.storage_index());
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");

        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::new(
                &mut owner,
                ModuleOp::get_concrete_op_info(),
                vec![owner_type, foreign_type],
                vec![],
                vec![],
                0,
            )
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            BasicBlock::new(&mut owner, None, vec![owner_type, foreign_type])
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::push_result(owner_fixture.operation, &owner, foreign_type)
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::insert_result(owner_fixture.operation, &owner, 0, foreign_type)
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            BasicBlock::push_argument(owner_fixture.block, &owner, foreign_type)
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            BasicBlock::insert_argument(owner_fixture.block, &owner, 0, foreign_type)
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            owner_fixture.value.set_type(&owner, foreign_type)
        });
        let mut rewriter = IRRewriter::<Recorder>::default();
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            rewriter.set_value_type(&owner, owner_fixture.value, foreign_type)
        });
        assert!(rewriter.get_listener().events.is_empty());
        assert!(!rewriter.is_modified());
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            FunctionType::get(&owner, vec![owner_type], vec![foreign_type])
        });

        let type_attr = TypeAttr::new(foreign_type);
        let error = expect_error(type_attr.verify(&owner));
        assert!(is_foreign_type_error(&error));

        let foreign_function_type = FunctionType::get(&foreign, vec![], vec![]);
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            FuncOp::new(
                &mut owner,
                "foreign_type_func".try_into().unwrap(),
                foreign_function_type,
            )
        });

        let owner_function_type = FunctionType::get(&owner, vec![], vec![]);
        let owner_function = FuncOp::new(
            &mut owner,
            "owner_type_func".try_into().unwrap(),
            owner_function_type,
        );
        let old_function_type = owner_function.get_attr_func_type(&owner).unwrap().clone();
        let owner_epoch = epoch(&owner);
        let error = expect_error(
            owner_function.try_set_attr_func_type(&owner, TypeAttr::new(foreign_type)),
        );
        assert!(is_foreign_type_error(&error));
        assert_eq!(epoch(&owner), owner_epoch);
        assert_eq!(
            *owner_function.get_attr_func_type(&owner).unwrap(),
            old_function_type
        );

        let owner_constant =
            ConstantOp::new(&mut owner, Box::new(TypeAttr::new(owner_type)) as AttrObj);
        let nested_foreign_attr: AttrObj =
            Box::new(VecAttr::new(vec![Box::new(DictAttr::new(vec![(
                "foreign_type".try_into().unwrap(),
                Box::new(TypeAttr::new(foreign_type)) as AttrObj,
            )])) as AttrObj]));
        let old_value = owner_constant.get_value(&owner);
        let owner_epoch = epoch(&owner);
        let error = expect_error(
            owner_constant.try_set_attr_builtin_constant_value(&owner, nested_foreign_attr),
        );
        assert!(is_foreign_type_error(&error));
        assert_eq!(epoch(&owner), owner_epoch);
        assert!(owner_constant.get_value(&owner) == old_value);
    }

    fn graph_snapshot(context: &Context, fixture: &ArenaFixture) -> ArenaGraphSnapshot {
        let operation = fixture.operation.deref(context);
        let operands = operation.operands().collect();
        let successors = operation.successors().collect();
        let regions = operation.regions().collect();
        let operation_next = operation.get_next();
        let operation_previous = operation.get_prev();
        let operation_container = operation.get_container();
        drop(operation);

        let region = fixture.region.deref(context);
        let region_parent = region.get_parent_op();
        let region_head = region.get_head();
        let region_tail = region.get_tail();
        drop(region);

        let block = fixture.block.deref(context);
        let block_next = block.get_next();
        let block_previous = block.get_prev();
        let block_container = block.get_container();
        drop(block);

        ArenaGraphSnapshot {
            operation_count: context.operations.len(),
            region_count: context.regions.len(),
            block_count: context.basic_blocks.len(),
            value_counter: context.value_counter.get(),
            use_counter: context.use_counter.get(),
            operands,
            successors,
            regions,
            operation_next,
            operation_previous,
            operation_container,
            region_parent,
            region_head,
            region_tail,
            block_next,
            block_previous,
            block_container,
            block_uses: fixture.block.uses(context),
            value_uses: fixture.value.uses(context),
        }
    }

    struct MustNotConvert;

    impl DialectConversion for MustNotConvert {
        fn can_convert_op(&self, _ctx: &Context, _op: Ptr<Operation>) -> bool {
            panic!("conversion callback ran before root preflight")
        }

        fn rewrite(
            &mut self,
            _ctx: &mut Context,
            _rewriter: &mut DialectConversionRewriter,
            _op: Ptr<Operation>,
            _operands_info: &OperandsInfo,
        ) -> crate::result::Result<()> {
            panic!("conversion rewrite ran before root preflight")
        }
    }

    #[test]
    fn read_only_access_preserves_epoch() {
        let mut ctx = Context::new();
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let before = epoch(&ctx);
        assert!(block.deref(&ctx).attributes.0.is_empty());
        assert_eq!(epoch(&ctx), before);
    }

    #[test]
    fn arena_mutation_and_failed_mutable_borrow_advance_epoch() {
        let mut ctx = Context::new();
        let block = BasicBlock::new(&mut ctx, None, vec![]);

        let before_mutation = epoch(&ctx);
        block.deref_mut(&ctx).attributes.0.clear();
        assert!(epoch(&ctx) > before_mutation);

        let read_guard = block.deref(&ctx);
        let before_failed_attempt = epoch(&ctx);
        assert!(block.try_deref_mut(&ctx).is_err());
        assert!(epoch(&ctx) > before_failed_attempt);
        drop(read_guard);

        let before_deallocation = epoch(&ctx);
        ArenaObj::dealloc(block, &mut ctx);
        assert!(epoch(&ctx) > before_deallocation);
    }

    #[test]
    fn type_and_uniqued_store_routes_advance_epoch() {
        let mut ctx = Context::new();

        let before_type = epoch(&ctx);
        let unit = UnitType::get(&ctx);
        assert!(epoch(&ctx) > before_type);

        let after_type_insert = epoch(&ctx);
        let _same_unit = UnitType::get(&ctx);
        assert_eq!(epoch(&ctx), after_type_insert);

        let before_type_mutation = epoch(&ctx);
        let handle: TypeHandle = unit.into();
        drop(handle.deref_mut(&ctx));
        assert!(epoch(&ctx) > before_type_mutation);

        let before_unique = epoch(&ctx);
        let _ = uniqued_any::save(&mut ctx, 7_u32);
        assert!(epoch(&ctx) > before_unique);

        let after_unique_insert = epoch(&ctx);
        let _ = uniqued_any::save(&mut ctx, 7_u32);
        assert_eq!(epoch(&ctx), after_unique_insert);
    }

    #[test]
    fn exhaustion_fails_closed_permanently() {
        let ctx = Context::new();
        ctx.ir_mutation_attempt_epoch.set(u64::MAX);
        assert!(ctx.ir_mutation_attempt_epoch().is_ok());

        ctx.note_ir_mutation_attempt();
        assert!(ctx.ir_mutation_attempt_epoch().is_err());
        ctx.note_ir_mutation_attempt();
        assert!(ctx.ir_mutation_attempt_epoch().is_err());
    }

    #[test]
    fn context_identity_sequence_never_wraps_or_reuses() {
        let mut next = NonZeroU64::new(u64::MAX);
        let final_identity = take_context_identity(&mut next).unwrap();
        assert_eq!(final_identity.0.get(), u64::MAX);
        assert_eq!(next, None);
        assert_eq!(
            take_context_identity(&mut next),
            Err(ContextIdentityExhausted)
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn concurrent_context_identity_allocation_is_unique_and_nonzero() {
        const THREADS: usize = 8;
        const CONTEXTS_PER_THREAD: usize = 64;

        let mut workers = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            workers.push(std::thread::spawn(|| {
                let mut identities = Vec::with_capacity(CONTEXTS_PER_THREAD);
                for _ in 0..CONTEXTS_PER_THREAD {
                    let context = Context::try_new().unwrap();
                    identities.push(context.identity.0.get());
                }
                identities
            }));
        }

        let mut identities = Vec::with_capacity(THREADS * CONTEXTS_PER_THREAD);
        for worker in workers {
            identities.extend(worker.join().unwrap());
        }
        assert!(identities.iter().all(|identity| *identity != 0));
        identities.sort_unstable();
        let original_len = identities.len();
        identities.dedup();
        assert_eq!(identities.len(), original_len);
    }

    #[test]
    fn equal_slot_operations_regions_blocks_and_values_reject_foreign_contexts() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");

        assert_eq!(owner_fixture.operation.idx, foreign_fixture.operation.idx);
        assert_eq!(owner_fixture.region.idx, foreign_fixture.region.idx);
        assert_eq!(owner_fixture.block.idx, foreign_fixture.block.idx);
        let DefiningEntity::Block(owner_value_block) = owner_fixture.value.defining_entity else {
            panic!("fixture value must be a block argument")
        };
        let DefiningEntity::Block(foreign_value_block) = foreign_fixture.value.defining_entity
        else {
            panic!("fixture value must be a block argument")
        };
        assert_eq!(owner_value_block.idx, foreign_value_block.idx);
        assert_ne!(owner_fixture.operation, foreign_fixture.operation);
        assert_ne!(owner_fixture.region, foreign_fixture.region);
        assert_ne!(owner_fixture.block, foreign_fixture.block);
        assert_ne!(owner_fixture.value, foreign_fixture.value);

        for error in [
            expect_error(owner_fixture.operation.try_deref(&foreign)),
            expect_error(owner_fixture.region.try_deref(&foreign)),
            expect_error(owner_fixture.block.try_deref(&foreign)),
            expect_error(owner_fixture.value.try_find_index(&foreign)),
        ] {
            assert!(is_foreign_context_error(&error));
        }

        let before = epoch(&foreign);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                owner_fixture.operation.deref(&foreign)
            }))
            .is_err()
        );
        assert_eq!(epoch(&foreign), before);
        let error = expect_error(owner_fixture.operation.try_deref_mut(&foreign));
        assert!(is_foreign_context_error(&error));
        assert_eq!(epoch(&foreign), before);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                owner_fixture.operation.deref_mut(&foreign)
            }))
            .is_err()
        );
        assert_eq!(epoch(&foreign), before);
        assert!(foreign_fixture.operation.try_deref(&foreign).is_ok());
        assert!(owner_fixture.operation.try_deref(&owner).is_ok());
    }

    #[test]
    fn equal_slot_composite_mutators_reject_before_changing_either_context() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");

        assert_eq!(owner_fixture.operation.idx, foreign_fixture.operation.idx);
        assert_eq!(owner_fixture.region.idx, foreign_fixture.region.idx);
        assert_eq!(owner_fixture.block.idx, foreign_fixture.block.idx);
        let owner_value_block = owner_fixture.value.defining_block().unwrap();
        let foreign_value_block = foreign_fixture.value.defining_block().unwrap();
        assert_eq!(owner_value_block.idx, foreign_value_block.idx);
        let owner_type: TypeHandle = IntegerType::get(&owner, 32, Signedness::Signless).into();

        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::push_result(foreign_fixture.operation, &owner, owner_type);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::insert_result(foreign_fixture.operation, &owner, 0, owner_type);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            BasicBlock::push_argument(foreign_fixture.block, &owner, owner_type);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            BasicBlock::insert_argument(foreign_fixture.block, &owner, 0, owner_type);
        });

        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::push_operand(owner_fixture.operation, &owner, foreign_fixture.value);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::push_operand(foreign_fixture.operation, &owner, owner_fixture.value);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::insert_operand(owner_fixture.operation, &owner, 0, foreign_fixture.value);
        });

        Operation::push_operand(owner_fixture.operation, &owner, owner_fixture.value);
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::replace_operand(owner_fixture.operation, &owner, 0, foreign_fixture.value);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            owner_fixture
                .value
                .replace_all_uses_with(&owner, &foreign_fixture.value);
        });

        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::push_successor(owner_fixture.operation, &owner, foreign_fixture.block);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::insert_successor(owner_fixture.operation, &owner, 0, foreign_fixture.block);
        });

        Operation::push_successor(owner_fixture.operation, &owner, owner_fixture.block);
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::replace_successor(owner_fixture.operation, &owner, 0, foreign_fixture.block);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            owner_fixture
                .block
                .retarget_all_preds_to(&owner, foreign_fixture.block);
        });

        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::new(
                &mut owner,
                ModuleOp::get_concrete_op_info(),
                vec![],
                vec![foreign_fixture.value],
                vec![foreign_fixture.block],
                0,
            );
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::add_region(foreign_fixture.operation, &mut owner);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Region::move_to_op(owner_fixture.region, foreign_fixture.operation, &owner);
        });

        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            owner_fixture
                .operation
                .insert_at_back(foreign_fixture.block, &owner);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            owner_fixture
                .operation
                .insert_after(&owner, foreign_fixture.operation);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            append_linked_list(owner_fixture.region, foreign_fixture.region, &owner);
        });
        assert_foreign_mutation_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            append_linked_list(foreign_fixture.region, owner_fixture.region, &owner);
        });
    }

    #[test]
    fn equal_slot_dialect_conversion_rejects_before_driver_or_callback_state() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        let mut conversion = MustNotConvert;
        assert_foreign_result_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            apply_dialect_conversion(&mut owner, &mut conversion, foreign_fixture.operation)
        });
    }

    #[test]
    fn equal_slot_dce_rejects_before_worklist_or_ir_state() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        assert_foreign_result_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            dce(foreign_fixture.operation, &mut owner)
        });
    }

    #[test]
    fn equal_slot_mem2reg_rejects_before_analysis_or_ir_state() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        let mut analyses = AnalysisManager::default();
        assert_foreign_result_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            mem2reg(foreign_fixture.operation, &mut owner, &mut analyses)
        });
        assert!(
            analyses
                .try_get_analysis::<DomInfo>(owner_fixture.operation)
                .is_none()
        );
    }

    #[test]
    fn equal_slot_simplify_cfg_rejects_before_worklist_or_ir_state() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        assert_foreign_result_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            simplify_cfg(foreign_fixture.operation, &mut owner)
        });
    }

    #[test]
    fn equal_slot_foreign_root_containment_is_rejected_without_changes() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        crate::linked_list::private::LinkedList::set_container(
            &mut *owner_fixture.operation.deref_mut(&owner),
            Some(foreign_fixture.block),
        );
        assert_foreign_result_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::try_preflight_tree(owner_fixture.operation, &owner)
        });
    }

    #[test]
    fn equal_slot_foreign_block_parent_is_rejected_without_changes() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        crate::linked_list::private::LinkedList::set_container(
            &mut *owner_fixture.block.deref_mut(&owner),
            Some(foreign_fixture.region),
        );
        assert_foreign_result_rejected!(owner, owner_fixture, foreign, foreign_fixture, {
            Operation::try_preflight_block_tree(owner_fixture.block, &owner)
        });
    }

    #[test]
    fn malformed_later_operand_is_rejected_before_any_use_drop() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let op = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![fixture.value, fixture.value],
            vec![],
            0,
        );
        op.deref_mut(&ctx).operands[1].use_uid = u64::MAX;
        let before = (
            graph_snapshot(&ctx, &fixture),
            op.deref(&ctx)
                .operands
                .iter()
                .map(|operand| (operand.r#use.get_def(), operand.use_uid))
                .collect::<Vec<_>>(),
        );

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| Operation::drop_all_uses(op, &ctx)))
                .is_err()
        );
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                op.deref(&ctx)
                    .operands
                    .iter()
                    .map(|operand| (operand.r#use.get_def(), operand.use_uid))
                    .collect::<Vec<_>>(),
            ) == before
        );
    }

    #[test]
    fn malformed_later_successor_is_rejected_before_any_use_drop() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let op = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![fixture.block, fixture.block],
            0,
        );
        op.deref_mut(&ctx).successors[1].use_uid = u64::MAX;
        let before = (
            graph_snapshot(&ctx, &fixture),
            op.deref(&ctx)
                .successors
                .iter()
                .map(|successor| (successor.r#use.get_def(), successor.use_uid))
                .collect::<Vec<_>>(),
        );

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| Operation::drop_all_uses(op, &ctx)))
                .is_err()
        );
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                op.deref(&ctx)
                    .successors
                    .iter()
                    .map(|successor| (successor.r#use.get_def(), successor.use_uid))
                    .collect::<Vec<_>>(),
            ) == before
        );
    }

    #[test]
    fn operation_erase_rejects_result_use_before_dropping_outgoing_uses() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let definition = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![ty],
            vec![fixture.value],
            vec![fixture.block],
            0,
        );
        let result = definition.deref(&ctx).get_result(0);
        let user = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![result],
            vec![],
            0,
        );
        let before = (
            graph_snapshot(&ctx, &fixture),
            definition.deref(&ctx).operands().collect::<Vec<_>>(),
            definition.deref(&ctx).successors().collect::<Vec<_>>(),
            result.uses(&ctx),
            user.deref(&ctx).operands().collect::<Vec<_>>(),
        );

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                Operation::erase(definition, &mut ctx)
            }))
            .is_err()
        );
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                definition.deref(&ctx).operands().collect::<Vec<_>>(),
                definition.deref(&ctx).successors().collect::<Vec<_>>(),
                result.uses(&ctx),
                user.deref(&ctx).operands().collect::<Vec<_>>(),
            ) == before
        );
    }

    #[test]
    fn block_erase_rejects_external_predecessor_before_dropping_uses() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let predecessor = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![fixture.value],
            vec![fixture.block],
            0,
        );
        let before = (
            graph_snapshot(&ctx, &fixture),
            predecessor.deref(&ctx).operands().collect::<Vec<_>>(),
            predecessor.deref(&ctx).successors().collect::<Vec<_>>(),
        );

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                BasicBlock::erase(fixture.block, &mut ctx)
            }))
            .is_err()
        );
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                predecessor.deref(&ctx).operands().collect::<Vec<_>>(),
                predecessor.deref(&ctx).successors().collect::<Vec<_>>(),
            ) == before
        );
    }

    #[test]
    fn region_erase_rejects_nested_external_result_use_before_dropping_uses() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let definition = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![ty],
            vec![fixture.value],
            vec![],
            0,
        );
        definition.insert_at_back(fixture.block, &ctx);
        let result = definition.deref(&ctx).get_result(0);
        let user = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![result],
            vec![],
            0,
        );
        let before = (
            graph_snapshot(&ctx, &fixture),
            definition.deref(&ctx).operands().collect::<Vec<_>>(),
            result.uses(&ctx),
            user.deref(&ctx).operands().collect::<Vec<_>>(),
        );

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                Operation::erase_region(fixture.operation, &mut ctx, 0)
            }))
            .is_err()
        );
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                definition.deref(&ctx).operands().collect::<Vec<_>>(),
                result.uses(&ctx),
                user.deref(&ctx).operands().collect::<Vec<_>>(),
            ) == before
        );
    }

    #[test]
    fn rewriter_replacement_rejects_malformed_later_result_before_events_or_mutation() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let definition = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![ty, ty],
            vec![fixture.value],
            vec![fixture.block],
            0,
        );
        let first_result = definition.deref(&ctx).get_result(0);
        let second_result = definition.deref(&ctx).get_result(1);
        let first_user = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![first_result],
            vec![],
            0,
        );
        let second_user = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![second_result],
            vec![],
            0,
        );
        second_user.deref_mut(&ctx).operands[0].use_uid = u64::MAX;
        let before = (
            graph_snapshot(&ctx, &fixture),
            definition.deref(&ctx).operands().collect::<Vec<_>>(),
            definition.deref(&ctx).successors().collect::<Vec<_>>(),
            first_result.uses(&ctx),
            second_result.uses(&ctx),
            first_user.deref(&ctx).operands().collect::<Vec<_>>(),
            second_user.deref(&ctx).operands[0].use_uid,
        );
        let mut rewriter = IRRewriter::<Recorder>::default();

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.replace_operation_with_values(
                    &mut ctx,
                    definition,
                    vec![fixture.value, fixture.value],
                )
            }))
            .is_err()
        );
        assert!(rewriter.get_listener().events.is_empty());
        assert!(!rewriter.is_modified());
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                definition.deref(&ctx).operands().collect::<Vec<_>>(),
                definition.deref(&ctx).successors().collect::<Vec<_>>(),
                first_result.uses(&ctx),
                second_result.uses(&ctx),
                first_user.deref(&ctx).operands().collect::<Vec<_>>(),
                second_user.deref(&ctx).operands[0].use_uid,
            ) == before
        );
    }

    #[test]
    fn rewriter_value_replacement_rejects_malformed_use_before_event_or_mutation() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let replacement_block = BasicBlock::new(&mut ctx, None, vec![ty]);
        let replacement = replacement_block.deref(&ctx).get_argument(0);
        let first_user = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![fixture.value],
            vec![],
            0,
        );
        let second_user = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![fixture.value],
            vec![],
            0,
        );
        second_user.deref_mut(&ctx).operands[0].use_uid = u64::MAX;
        let before = (
            graph_snapshot(&ctx, &fixture),
            replacement.uses(&ctx),
            first_user.deref(&ctx).operands().collect::<Vec<_>>(),
            second_user.deref(&ctx).operands[0].use_uid,
        );
        let mut rewriter = IRRewriter::<Recorder>::default();

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.replace_value_uses_with(&ctx, fixture.value, replacement)
            }))
            .is_err()
        );
        assert!(rewriter.get_listener().events.is_empty());
        assert!(!rewriter.is_modified());
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                replacement.uses(&ctx),
                first_user.deref(&ctx).operands().collect::<Vec<_>>(),
                second_user.deref(&ctx).operands[0].use_uid,
            ) == before
        );
    }

    #[test]
    fn rewriter_rejects_same_or_nested_replacement_definitions_before_events() {
        let mut ctx = Context::try_new().unwrap();
        let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let root = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![ty],
            vec![],
            vec![],
            1,
        );
        let region = root.deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_front(region, &ctx);
        let nested = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![ty],
            vec![],
            vec![],
            0,
        );
        nested.insert_at_front(block, &ctx);
        let nested_result = nested.deref(&ctx).get_result(0);
        let before = (
            ctx.operations.len(),
            ctx.regions.len(),
            ctx.basic_blocks.len(),
            ctx.value_counter.get(),
            ctx.use_counter.get(),
            epoch(&ctx),
            root.deref(&ctx).regions().collect::<Vec<_>>(),
            region.deref(&ctx).get_head(),
            block.deref(&ctx).get_head(),
            nested_result.uses(&ctx),
        );
        let mut rewriter = IRRewriter::<Recorder>::default();

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.replace_operation_with_values(&mut ctx, root, vec![nested_result])
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.replace_operation(&mut ctx, root, root)
            }))
            .is_err()
        );
        assert!(rewriter.get_listener().events.is_empty());
        assert!(!rewriter.is_modified());
        assert!(
            (
                ctx.operations.len(),
                ctx.regions.len(),
                ctx.basic_blocks.len(),
                ctx.value_counter.get(),
                ctx.use_counter.get(),
                epoch(&ctx),
                root.deref(&ctx).regions().collect::<Vec<_>>(),
                region.deref(&ctx).get_head(),
                block.deref(&ctx).get_head(),
                nested_result.uses(&ctx),
            ) == before
        );
    }

    #[test]
    fn inserter_rejects_foreign_operation_and_block_destinations_before_events() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        let operation = Operation::new(
            &mut owner,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        let block = BasicBlock::new(&mut owner, None, vec![]);
        let owner_before = graph_snapshot(&owner, &owner_fixture);
        let foreign_before = graph_snapshot(&foreign, &foreign_fixture);

        let mut op_inserter =
            IRInserter::<Recorder>::new(OpInsertionPoint::AtBlockEnd(foreign_fixture.block));
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                op_inserter.insert_operation(&owner, operation)
            }))
            .is_err()
        );
        assert!(op_inserter.get_listener().events.is_empty());
        assert!(!op_inserter.is_modified());

        let mut block_inserter = IRInserter::<Recorder>::default();
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                block_inserter.insert_block(
                    &owner,
                    BlockInsertionPoint::AtRegionEnd(foreign_fixture.region),
                    block,
                )
            }))
            .is_err()
        );
        assert!(block_inserter.get_listener().events.is_empty());
        assert!(!block_inserter.is_modified());
        assert!(graph_snapshot(&owner, &owner_fixture) == owner_before);
        assert!(graph_snapshot(&foreign, &foreign_fixture) == foreign_before);
        assert!(!operation.is_linked(&owner));
        assert!(!block.is_linked(&owner));
    }

    #[test]
    fn rewriter_rejects_moves_and_inline_into_own_subtree_before_events() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let root = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        root.insert_at_back(fixture.block, &ctx);
        let root_region = root.deref(&ctx).get_region(0);
        let root_block = BasicBlock::new(&mut ctx, None, vec![]);
        root_block.insert_at_front(root_region, &ctx);
        let nested = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        nested.insert_at_front(root_block, &ctx);
        let nested_region = nested.deref(&ctx).get_region(0);
        let before = (
            graph_snapshot(&ctx, &fixture),
            root.deref(&ctx).get_parent_block(),
            root_region.deref(&ctx).get_head(),
            root_block.deref(&ctx).get_head(),
            nested_region.deref(&ctx).get_head(),
        );
        let mut rewriter = IRRewriter::<Recorder>::default();

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.move_operation(&ctx, root, OpInsertionPoint::BeforeOperation(root))
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.move_block(
                    &ctx,
                    root_block,
                    BlockInsertionPoint::BeforeBlock(root_block),
                )
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.move_operation(&ctx, root, OpInsertionPoint::AtBlockEnd(root_block))
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.move_block(
                    &ctx,
                    root_block,
                    BlockInsertionPoint::AtRegionEnd(nested_region),
                )
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.inline_region(
                    &ctx,
                    root_region,
                    BlockInsertionPoint::AtRegionEnd(nested_region),
                )
            }))
            .is_err()
        );
        assert!(rewriter.get_listener().events.is_empty());
        assert!(!rewriter.is_modified());
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                root.deref(&ctx).get_parent_block(),
                root_region.deref(&ctx).get_head(),
                root_block.deref(&ctx).get_head(),
                nested_region.deref(&ctx).get_head(),
            ) == before
        );
    }

    #[test]
    fn split_rejects_position_in_another_block_before_allocation_or_events() {
        let mut ctx = Context::try_new().unwrap();
        let fixture = arena_fixture(&mut ctx, "owner");
        let other = BasicBlock::new(&mut ctx, None, vec![]);
        other.insert_at_back(fixture.region, &ctx);
        let before = (
            graph_snapshot(&ctx, &fixture),
            other.deref(&ctx).get_container(),
            other.deref(&ctx).get_prev(),
            other.deref(&ctx).get_next(),
        );
        let mut rewriter = IRRewriter::<Recorder>::default();

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.split_block(
                    &mut ctx,
                    fixture.block,
                    OpInsertionPoint::AtBlockStart(other),
                    None,
                )
            }))
            .is_err()
        );
        assert!(rewriter.get_listener().events.is_empty());
        assert!(!rewriter.is_modified());
        assert!(
            (
                graph_snapshot(&ctx, &fixture),
                other.deref(&ctx).get_container(),
                other.deref(&ctx).get_prev(),
                other.deref(&ctx).get_next(),
            ) == before
        );
    }

    #[test]
    fn type_change_authenticates_new_type_before_listener_or_value_mutation() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        let mut foreign_type: TypeHandle = UnitType::get(&foreign).into();
        for width in 1..128 {
            foreign_type = IntegerType::get(&foreign, width, Signedness::Unsigned).into();
        }
        let owner_before = graph_snapshot(&owner, &owner_fixture);
        let foreign_before = graph_snapshot(&foreign, &foreign_fixture);
        let old_type = owner_fixture.value.get_type(&owner);
        let mut rewriter = IRRewriter::<Recorder>::default();

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                rewriter.set_value_type(&owner, owner_fixture.value, foreign_type)
            }))
            .is_err()
        );
        assert!(rewriter.get_listener().events.is_empty());
        assert!(!rewriter.is_modified());
        assert!(graph_snapshot(&owner, &owner_fixture) == owner_before);
        assert!(graph_snapshot(&foreign, &foreign_fixture) == foreign_before);
        assert!(owner_fixture.value.get_type(&owner) == old_type);
    }

    #[test]
    fn result_returning_queries_report_foreign_and_dangling_user_pointers() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = arena_fixture(&mut owner, "owner");
        let foreign_fixture = arena_fixture(&mut foreign, "foreign");
        assert_eq!(owner_fixture.operation.idx, foreign_fixture.operation.idx);

        Operation::push_operand(owner_fixture.operation, &owner, owner_fixture.value);
        let value_use = owner_fixture.operation.deref(&owner).get_operand_as_use(0);
        let error = expect_error(value_use.try_find_index(&foreign));
        assert!(is_foreign_context_error(&error));

        Operation::push_successor(owner_fixture.operation, &owner, owner_fixture.block);
        let successor_use = owner_fixture
            .operation
            .deref(&owner)
            .get_successor_as_use(0);
        let error = expect_error(successor_use.try_find_index(&foreign));
        assert!(is_foreign_context_error(&error));

        let error = expect_error(verify_operation(owner_fixture.operation, &foreign));
        assert!(is_foreign_context_error(&error));

        let dangling = Operation::new(
            &mut owner,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ArenaObj::dealloc(dangling, &mut owner);
        let dangling_value_use = Use::<Value> {
            user_op: dangling,
            use_uid: u64::MAX,
            _dummy: PhantomData,
        };
        let error = expect_error(dangling_value_use.try_find_index(&owner));
        assert!(is_dangling_pointer_error(&error));
        let dangling_successor_use = Use::<Ptr<BasicBlock>> {
            user_op: dangling,
            use_uid: u64::MAX,
            _dummy: PhantomData,
        };
        let error = expect_error(dangling_successor_use.try_find_index(&owner));
        assert!(is_dangling_pointer_error(&error));
        let error = expect_error(verify_operation(dangling, &owner));
        assert!(is_dangling_pointer_error(&error));
    }

    #[test]
    fn foreign_deallocation_rejects_before_indexing_or_mutating_either_context() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_block = BasicBlock::new(&mut owner, None, vec![]);
        let foreign_block = BasicBlock::new(&mut foreign, None, vec![]);
        assert_eq!(owner_block.idx, foreign_block.idx);

        let owner_before = epoch(&owner);
        let foreign_before = epoch(&foreign);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                ArenaObj::dealloc(owner_block, &mut foreign)
            }))
            .is_err()
        );
        assert_eq!(epoch(&owner), owner_before);
        assert_eq!(epoch(&foreign), foreign_before);
        assert!(owner_block.try_deref(&owner).is_ok());
        assert!(foreign_block.try_deref(&foreign).is_ok());
    }

    #[test]
    fn stale_pointer_remains_dangling_after_same_context_reallocation() {
        let mut context = Context::try_new().unwrap();
        let stale = BasicBlock::new(&mut context, None, vec![]);
        ArenaObj::dealloc(stale, &mut context);

        let error = expect_error(stale.try_deref(&context));
        assert!(error.err.downcast_ref::<DanglingPtrDerefError>().is_some());
        let replacement = BasicBlock::new(&mut context, None, vec![]);
        assert_ne!(stale, replacement);
        let error = expect_error(stale.try_deref(&context));
        assert!(error.err.downcast_ref::<DanglingPtrDerefError>().is_some());
        assert!(replacement.try_deref(&context).is_ok());
    }
}
