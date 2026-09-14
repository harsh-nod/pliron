// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

//! Optional per-rewriter mutation observations, independent of scheduling listeners.
//!
//! Notifications expose live objects at the mutation boundary, not a persistent
//! pointer-based provenance format. Observers must copy any needed facts before
//! erasure. They cannot veto mutations; bounded observers can record a sticky
//! failure and require their caller to discard the candidate after the pass.
//!
//! # Callback Timing and Scope
//!
//! Insertion callbacks run after insertion and after the legacy insertion
//! listener. Erasure, unlinking, value-use replacement and type-change callbacks
//! run before their corresponding mutation and legacy listener callback. Recursive
//! erasure reports descendants before their parent, so a parent callback may see
//! an already emptied subtree. OperationReplaced runs before use replacement and
//! erasure, but after any configured replacement-location copying; it has no
//! corresponding legacy operation-replacement notification.
//!
//! Moving a linked operation emits unlink followed by insertion notifications.
//! Inserting a prebuilt operation subtree reports only its root, not each nested
//! operation, block or region. These hooks are not an exhaustive audit of Context
//! mutations: direct IR APIs and other rewriters do not emit these observations.
//! Callers requiring complete coverage must reconcile the resulting graph.
//!
//! Callbacks may observe temporarily invalid IR. A callback panic follows the
//! build's panic strategy; unwinding does not roll back preceding IR changes or
//! listener calls. Callers must discard or independently reconcile the candidate
//! before further use.

use crate::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    operation::Operation,
    region::Region,
    r#type::TypeHandle,
    value::Value,
};

/// A mutation observed through an [IRRewriter](super::rewriter::IRRewriter).
/// Insertion also covers reinsertion of an existing, unlinked object.
#[derive(Clone, Copy, Debug)]
pub enum RewriteEvent {
    OperationInserted(Ptr<Operation>),
    BlockInserted(Ptr<BasicBlock>),
    OperationErased(Ptr<Operation>),
    BlockErased(Ptr<BasicBlock>),
    RegionErased(Ptr<Region>),
    ValueReplaced {
        old: Value,
        new: Value,
    },
    /// Explicit operation replacement, including zero-result operations.
    OperationReplaced {
        old: Ptr<Operation>,
        new: Ptr<Operation>,
    },
    OperationUnlinked(Ptr<Operation>),
    BlockUnlinked(Ptr<BasicBlock>),
    ValueTypeChanged {
        value: Value,
        old_type: TypeHandle,
        new_type: TypeHandle,
    },
}

/// An occurrence-sensitive boundary inside a transform about to mutate live IR.
/// These opt-in notifications do not replace the legacy RewriteEvent stream.
/// They are observations of the running transform, not a mutation veto or proof.
#[derive(Clone, Copy, Debug)]
pub enum RewriteOccurrenceEvent {
    /// Before destructive removal of all operand and successor uses in this
    /// unreachable block. Later erasure callbacks can see empty operand lists.
    UnreachableBlockDroppingUses { block: Ptr<BasicBlock> },
    /// Before removing this block argument and every incoming argument slot.
    /// The argument and all predecessor occurrences are still live.
    BlockArgumentErasing {
        block: Ptr<BasicBlock>,
        argument: Value,
    },
    /// Before forwarding the predecessor's sole edge arguments, erasing its
    /// terminator and moving the successor's operations into the predecessor.
    SuccessorBlockMerging {
        predecessor: Ptr<BasicBlock>,
        successor: Ptr<BasicBlock>,
        terminator: Ptr<Operation>,
    },
    /// After inserting the new unconditional branch, before replacing the old
    /// branch. The ordinal identifies an occurrence, even for duplicate targets.
    BranchSuccessorSelected {
        old: Ptr<Operation>,
        new: Ptr<Operation>,
        successor: usize,
    },
}

/// An owned, opt-in observer. Existing listener event streams are unchanged.
///
/// The context is read-only. Implementations must not mutate IR through interior
/// mutability. Observations are not emitted for mutations made outside the
/// rewriter, so callers must independently reconcile pre/post graph coverage.
pub trait RewriteObserver:
    Send + Sync + core::panic::UnwindSafe + core::panic::RefUnwindSafe
{
    fn observe(&mut self, ctx: &Context, event: RewriteEvent);

    /// Opt in only when occurrence-sensitive transform boundaries are needed.
    /// Legacy observers perform no occurrence census and receive no such events.
    fn observes_occurrences(&self) -> bool {
        false
    }

    /// Observes a boundary only when observes_occurrences returned true.
    /// Copy needed facts here: a subsequent mutation can invalidate these objects.
    fn observe_occurrence(&mut self, _ctx: &Context, _event: RewriteOccurrenceEvent) {}
}
