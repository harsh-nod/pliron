// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

//! Utilities to work with [SymbolTableInterface] Ops.

use alloc::{
    boxed::Box,
    string::{String, ToString},
};
use thiserror::Error;

use crate::{
    arg_err, arg_error,
    builtin::op_interfaces::{SymbolOpInterface, SymbolTableInterface},
    context::{Context, Ptr},
    graph::walkers::{
        IRNode, WALKCONFIG_PREORDER_FORWARD,
        interruptible::{WalkResult, immutable::walk_region, walk_advance, walk_skip},
    },
    identifier::Identifier,
    irbuild::{
        inserter::{IRInserter, Inserter, OpInsertionPoint},
        listener::DummyListener,
    },
    linked_list::{ContainsLinkedList, try_preflight_linked_list},
    location::Located,
    op::op_cast,
    operation::Operation,
    printable::Printable,
    result::Result,
    utils::table::HMap,
};

/// A utility to efficiently lookup and update [Symbol](SymbolOpInterface)s
/// in a [SymbolTableInterface] Op. Similar to its counterpart in MLIR,
/// inserting into and erasing from this [SymbolTable] will also insert and
/// erase from the Operation given to it at construction.
pub struct SymbolTable {
    symbol_table_op: Box<dyn SymbolTableInterface>,
    symbol_table: HMap<Identifier, Box<dyn SymbolOpInterface>>,
}

#[derive(Error, Debug)]

pub enum SymbolTableErr {
    #[error("Op does not implement SymbolTableInterface")]
    DoesNotImplementSymbolTableInterface,
    #[error("Symbol redefined: {0}")]
    SymbolRedefined(String),
    #[error("Op does not implement SymbolOpInterface")]
    DoesNotImplementSymbolOpInterface,
    #[error("Symbol Op cannot be inserted as it is already in another symbol table")]
    SymbolOpAlreadyInAnotherTable(String),
    #[error("Insertion point must already be inside the same symbol table")]
    InsertionPointNotInSameTable,
    #[error("Symbol op does not belong to the symbol table op")]
    SymbolOpNotInTableOp,
}

fn try_block_parent_op(
    block: Ptr<crate::basic_block::BasicBlock>,
    ctx: &Context,
) -> Result<Option<Ptr<Operation>>> {
    drop(block.try_deref(ctx)?);
    try_preflight_linked_list(block, ctx)?;
    let Some(region) = block.try_deref(ctx)?.get_parent_region() else {
        return Ok(None);
    };

    try_preflight_linked_list(region, ctx)?;
    assert!(
        region
            .try_deref(ctx)?
            .iter(ctx)
            .any(|candidate| candidate == block)
    );
    let parent_op = region.try_deref(ctx)?.get_parent_op();
    let parent_op_ref = parent_op.try_deref(ctx)?;
    assert!(parent_op_ref.regions().any(|candidate| candidate == region));
    Ok(Some(parent_op))
}

fn try_operation_parent_op(op: Ptr<Operation>, ctx: &Context) -> Result<Option<Ptr<Operation>>> {
    let Some(block) = op.try_deref(ctx)?.get_parent_block() else {
        return Ok(None);
    };

    try_preflight_linked_list(block, ctx)?;
    assert!(
        block
            .try_deref(ctx)?
            .iter(ctx)
            .any(|candidate| candidate == op)
    );
    try_block_parent_op(block, ctx)
}

fn try_insertion_block(
    insert_pt: OpInsertionPoint,
    ctx: &Context,
) -> Result<Option<Ptr<crate::basic_block::BasicBlock>>> {
    let block = match insert_pt {
        OpInsertionPoint::AtBlockStart(block) | OpInsertionPoint::AtBlockEnd(block) => Some(block),
        OpInsertionPoint::AfterOperation(op) | OpInsertionPoint::BeforeOperation(op) => {
            try_operation_parent_op(op, ctx)?;
            op.try_deref(ctx)?.get_parent_block()
        }
        OpInsertionPoint::Unset => None,
    };
    if let Some(block) = block {
        try_block_parent_op(block, ctx)?;
    }
    Ok(block)
}

impl SymbolTable {
    /// Create a new [SymbolTable] from a [SymbolTableInterface] Op.
    pub fn new(ctx: &Context, symbol_table_op: Box<dyn SymbolTableInterface>) -> Self {
        Operation::try_preflight_tree(symbol_table_op.get_operation(), ctx)
            .expect("Symbol table operation preflight failed");

        // Go through the all operations in symbol_table_op and build a table.
        let mut symbol_table = HMap::<Identifier, Box<dyn SymbolOpInterface>>::default();
        let table_ops_block = symbol_table_op.get_body(ctx, 0);
        for op in table_ops_block.deref(ctx).iter(ctx) {
            if let Some(sym_op) =
                op_cast::<dyn SymbolOpInterface>(Operation::get_op_dyn(op, ctx).as_ref())
            {
                let sym = sym_op.get_symbol_name(ctx);
                if let Some(prev_op) =
                    symbol_table.insert(sym.clone(), dyn_clone::clone_box(sym_op))
                {
                    // It's the job of the verifier to have caught this. So we panic.
                    panic!(
                        "Symbol {} defined at {} was previously defined at {}",
                        sym,
                        symbol_table_op.loc(ctx).disp(ctx),
                        prev_op.loc(ctx).disp(ctx)
                    );
                }
            }
        }

        Self {
            symbol_table_op,
            symbol_table,
        }
    }

    /// Insert a new [Symbol](SymbolOpInterface) into the [SymbolTable].
    /// If an insertion point is provided, that point must be inside self.symbol_table_op.
    /// If no insertion point is provided, the symbol op will be inserted at the end
    /// (before the terminator if one exists). symbol_op's parent must be either `None`,
    /// or the same as self.symbol_table_op.
    pub fn insert(
        &mut self,
        ctx: &Context,
        symbol_op: Box<dyn SymbolOpInterface>,
        insert_pt: Option<OpInsertionPoint>,
    ) -> Result<()> {
        let symbol_table_op = self.symbol_table_op.get_operation();
        let symbol_op_ptr = symbol_op.get_operation();

        // Complete every pointer and structural check before symbol-name cloning,
        // map growth, listener notification, or IR mutation.
        Operation::try_preflight_tree(symbol_table_op, ctx)?;
        Operation::try_preflight_tree(symbol_op_ptr, ctx)?;
        try_operation_parent_op(symbol_table_op, ctx)?;
        let symbol_op_parent = try_operation_parent_op(symbol_op_ptr, ctx)?;
        let insertion_block = match insert_pt {
            Some(insert_pt) => try_insertion_block(insert_pt, ctx)?,
            None => None,
        };

        let symbol_name = symbol_op.get_symbol_name(ctx);
        if let Some(prev_op) = self.symbol_table.get(&symbol_name) {
            return arg_err!(
                symbol_op.loc(ctx),
                arg_error!(
                    prev_op.loc(ctx),
                    SymbolTableErr::SymbolRedefined(symbol_name.to_string())
                )
            );
        }

        if let Some(symbol_op_parent) = symbol_op_parent {
            if symbol_table_op != symbol_op_parent {
                return arg_err!(
                    symbol_op_ptr.deref(ctx).loc(),
                    SymbolTableErr::SymbolOpAlreadyInAnotherTable(symbol_name.to_string())
                );
            }
            // Nothing more to do.
        } else if let Some(insert_pt) = insert_pt {
            let insert_pt_parent = match insertion_block {
                Some(block) => try_block_parent_op(block, ctx)?,
                None => None,
            };
            match insert_pt_parent {
                Some(parent) if parent == symbol_table_op => {
                    let mut inserter = IRInserter::<DummyListener>::new(insert_pt);
                    inserter.insert_operation(ctx, symbol_op_ptr);
                }
                _ => {
                    return arg_err!(
                        symbol_op_ptr.deref(ctx).loc(),
                        SymbolTableErr::InsertionPointNotInSameTable
                    );
                }
            }
        } else {
            // Insert at the end, before the terminator if one exists.
            let block = self.symbol_table_op.get_body(ctx, 0);
            let terminator = block.deref(ctx).get_terminator(ctx);
            match terminator {
                Some(terminator) => symbol_op_ptr.insert_before(ctx, terminator),
                None => symbol_op_ptr.insert_at_back(block, ctx),
            }
        }

        let previous = self.symbol_table.insert(symbol_name, symbol_op);
        assert!(previous.is_none(), "Symbol table changed during insertion");
        Ok(())
    }

    /// Remove `symbol_op` from this [SymbolTable].
    /// The operation is unlinked from its parent, but is not erased.
    pub fn remove(&mut self, ctx: &Context, symbol_op: Box<dyn SymbolOpInterface>) -> Result<()> {
        let symbol_op_opr = symbol_op.get_operation();
        let symbol_table_op = self.symbol_table_op.get_operation();
        Operation::try_preflight_tree(symbol_table_op, ctx)?;
        Operation::try_preflight_tree(symbol_op_opr, ctx)?;
        try_operation_parent_op(symbol_table_op, ctx)?;
        let symbol_parent = try_operation_parent_op(symbol_op_opr, ctx)?;
        if symbol_parent != Some(symbol_table_op) {
            return arg_err!(symbol_op.loc(ctx), SymbolTableErr::SymbolOpNotInTableOp);
        }

        let symbol = symbol_op.get_symbol_name(ctx);
        if !matches!(
            self.symbol_table.get(&symbol),
            Some(registered) if registered.get_operation() == symbol_op_opr
        ) {
            return arg_err!(symbol_op.loc(ctx), SymbolTableErr::SymbolOpNotInTableOp);
        }
        self.symbol_table.remove(&symbol);

        symbol_op_opr.unlink(ctx);

        Ok(())
    }

    /// Remove `symbol_op` from this [SymbolTable] and erase it.
    /// This will panic if `symbol_op` has SSA uses.
    pub fn erase(
        &mut self,
        ctx: &mut Context,
        symbol_op: Box<dyn SymbolOpInterface>,
    ) -> Result<()> {
        Operation::try_preflight_erase_tree(symbol_op.get_operation(), ctx)?;
        self.remove(ctx, symbol_op.clone())?;
        Operation::erase(symbol_op.get_operation(), ctx);
        Ok(())
    }

    /// Get the [Symbol](SymbolOpInterface) with the given name.
    pub fn lookup(&self, symbol_name: &Identifier) -> Option<Box<dyn SymbolOpInterface>> {
        self.symbol_table.get(symbol_name).cloned()
    }
}

pub type SymbolTableWalkerCallback<State> =
    fn(&Context, &mut State, Ptr<Operation>) -> WalkResult<()>;

/// Walk all (nested) operations within the region of `symbol_table_op`,
/// without traversing into any nested symbol tables.
pub fn walk_symbol_table<State>(
    symbol_table_op: Box<dyn SymbolTableInterface>,
    ctx: &Context,
    state: &mut State,
    callback: SymbolTableWalkerCallback<State>,
) {
    struct StateWithCallback<'a, State> {
        callback: SymbolTableWalkerCallback<State>,
        state: &'a mut State,
    }

    fn skip_nested_symbol_tables_callback<State>(
        ctx: &Context,
        state: &mut StateWithCallback<State>,
        node: IRNode,
    ) -> WalkResult<()> {
        if let IRNode::Operation(opr) = node {
            let op = Operation::get_op_dyn(opr, ctx);
            if op_cast::<dyn SymbolTableInterface>(op.as_ref()).is_some() {
                return walk_skip();
            }
            (state.callback)(ctx, state.state, opr)
        } else {
            walk_advance()
        }
    }

    let _ = walk_region(
        ctx,
        &mut StateWithCallback { callback, state },
        &WALKCONFIG_PREORDER_FORWARD,
        symbol_table_op.get_region(ctx),
        skip_nested_symbol_tables_callback,
    );
}

/// A collection of [SymbolTable]s, to simplify algorithms that run
/// recursively on nested [SymbolTable]s. The actual tables are constructed lazily.
/// See MLIR's [SymbolTableCollection](https://mlir.llvm.org/doxygen/classmlir_1_1SymbolTableCollection.html).
#[derive(Default)]
pub struct SymbolTableCollection {
    symbol_tables: HMap<Ptr<Operation>, SymbolTable>,
}

/// Returns the nearest symbol table from a given operation `from`
pub fn nearest_symbol_table(
    ctx: &Context,
    from: Ptr<Operation>,
) -> Option<Box<dyn SymbolTableInterface>> {
    let mut op = from;
    loop {
        if let Some(symbol_table) =
            op_cast::<dyn SymbolTableInterface>(Operation::get_op_dyn(op, ctx).as_ref())
        {
            return Some(dyn_clone::clone_box(symbol_table));
        }
        op = op.deref(ctx).get_parent_op(ctx)?;
    }
}

impl SymbolTableCollection {
    /// Create a new empty [SymbolTableCollection].
    pub fn new() -> Self {
        Self::default()
    }

    /// Lookup symbol in the given [SymbolTableInterface] Op.
    /// A symbol table is computed on demand and cached in the collection.
    pub fn lookup_symbol_in_table(
        &mut self,
        ctx: &Context,
        symbol_table_op: Box<dyn SymbolTableInterface>,
        symbol_name: &Identifier,
    ) -> Option<Box<dyn SymbolOpInterface>> {
        self.get_symbol_table(ctx, symbol_table_op)
            .lookup(symbol_name)
    }

    /// Get the symbol table for the symbol table operation, constructing it if necessary.
    pub fn get_symbol_table(
        &mut self,
        ctx: &Context,
        symbol_table_op: Box<dyn SymbolTableInterface>,
    ) -> &mut SymbolTable {
        Operation::try_preflight_tree(symbol_table_op.get_operation(), ctx)
            .expect("Symbol table operation preflight failed");
        self.symbol_tables
            .entry(symbol_table_op.get_operation())
            .or_insert_with(|| SymbolTable::new(ctx, symbol_table_op))
    }

    /// Returns the [SymbolOpInterface] Op registered with the given symbol within the
    /// closest [SymbolTableInterface] parent Op of, or including, 'from'.
    pub fn lookup_symbol_in_nearest_table(
        &mut self,
        ctx: &Context,
        from: Ptr<Operation>,
        symbol_name: &Identifier,
    ) -> Option<Box<dyn SymbolOpInterface>> {
        let symbol_table = nearest_symbol_table(ctx, from)?;
        self.lookup_symbol_in_table(ctx, symbol_table, symbol_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builtin::{
            op_interfaces::SingleBlockRegionInterface,
            ops::{FuncOp, ModuleOp},
            types::FunctionType,
        },
        context::{Context, ForeignContextPtrDerefError},
        identifier::Identifier,
        linked_list::LinkedList,
        op::Op,
    };
    use alloc::vec;
    use core::panic::AssertUnwindSafe;

    #[test]
    fn test_symbol_table() -> Result<()> {
        let ctx = &mut Context::new();
        let module_op = ModuleOp::new(ctx, Identifier::try_from("module").unwrap());
        let mut symbol_table = SymbolTable::new(ctx, Box::new(module_op));

        let fty = FunctionType::get(ctx, vec![], vec![]);
        let f1 = FuncOp::new(ctx, "f1".try_into().unwrap(), fty);

        symbol_table.insert(ctx, Box::new(f1), None)?;
        assert!(
            f1.get_operation().deref(ctx).get_parent_op(ctx) == Some(module_op.get_operation())
        );
        let f1_lookedup = symbol_table.lookup(&"f1".try_into().unwrap()).unwrap();
        assert!(f1_lookedup.get_operation() == f1.get_operation());

        let f2 = FuncOp::new(ctx, "f2".try_into().unwrap(), fty);
        symbol_table.insert(
            ctx,
            Box::new(f2),
            Some(OpInsertionPoint::BeforeOperation(f1.get_operation())),
        )?;
        assert!(
            f2.get_operation().deref(ctx).get_parent_op(ctx) == Some(module_op.get_operation())
        );
        let f2_lookedup = symbol_table.lookup(&"f2".try_into().unwrap()).unwrap();
        assert!(f2_lookedup.get_operation() == f2.get_operation());

        symbol_table.remove(ctx, Box::new(f1))?;
        assert!(symbol_table.lookup(&"f1".try_into().unwrap()).is_none());
        assert!(f1.get_operation().deref(ctx).get_parent_op(ctx).is_none());

        Ok(())
    }

    #[test]
    fn foreign_equal_slot_insertion_point_leaves_symbol_table_and_ir_unchanged() {
        #[derive(Eq, PartialEq)]
        struct Snapshot {
            operation_count: usize,
            region_count: usize,
            block_count: usize,
            value_counter: u64,
            use_counter: u64,
            epoch: u64,
            module_body_head: Option<Ptr<Operation>>,
            module_body_tail: Option<Ptr<Operation>>,
            function_parent: Option<Ptr<crate::basic_block::BasicBlock>>,
            function_prev: Option<Ptr<Operation>>,
            function_next: Option<Ptr<Operation>>,
        }

        fn snapshot(ctx: &Context, module: ModuleOp, function: FuncOp) -> Snapshot {
            let body = module.get_body(ctx, 0);
            let body_ref = body.deref(ctx);
            let module_body_head = body_ref.get_head();
            let module_body_tail = body_ref.get_tail();
            drop(body_ref);
            let function_ref = function.get_operation().deref(ctx);
            Snapshot {
                operation_count: ctx.operations.len(),
                region_count: ctx.regions.len(),
                block_count: ctx.basic_blocks.len(),
                value_counter: ctx.value_counter.get(),
                use_counter: ctx.use_counter.get(),
                epoch: ctx.ir_mutation_attempt_epoch().unwrap().value(),
                module_body_head,
                module_body_tail,
                function_parent: function_ref.get_parent_block(),
                function_prev: function_ref.get_prev(),
                function_next: function_ref.get_next(),
            }
        }

        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_module = ModuleOp::new(&mut owner, "owner".try_into().unwrap());
        let foreign_module = ModuleOp::new(&mut foreign, "foreign".try_into().unwrap());
        let owner_type = FunctionType::get(&owner, vec![], vec![]);
        let foreign_type = FunctionType::get(&foreign, vec![], vec![]);
        let owner_function = FuncOp::new(&mut owner, "candidate".try_into().unwrap(), owner_type);
        let foreign_function =
            FuncOp::new(&mut foreign, "foreign_fn".try_into().unwrap(), foreign_type);
        assert_eq!(
            owner_function.get_operation().idx,
            foreign_function.get_operation().idx
        );

        let mut symbol_table = SymbolTable::new(&owner, Box::new(owner_module));
        let owner_before = snapshot(&owner, owner_module, owner_function);
        let foreign_before = snapshot(&foreign, foreign_module, foreign_function);
        let table_len = symbol_table.symbol_table.len();
        let table_capacity = symbol_table.symbol_table.capacity();

        let error = symbol_table
            .insert(
                &owner,
                Box::new(owner_function),
                Some(OpInsertionPoint::BeforeOperation(
                    foreign_function.get_operation(),
                )),
            )
            .expect_err("foreign insertion point must be rejected");
        assert!(
            error
                .err
                .downcast_ref::<ForeignContextPtrDerefError>()
                .is_some()
        );

        assert!(snapshot(&owner, owner_module, owner_function) == owner_before);
        assert!(snapshot(&foreign, foreign_module, foreign_function) == foreign_before);
        assert_eq!(symbol_table.symbol_table.len(), table_len);
        assert_eq!(symbol_table.symbol_table.capacity(), table_capacity);
        assert!(
            symbol_table
                .lookup(&"candidate".try_into().unwrap())
                .is_none()
        );
    }

    #[test]
    fn erase_with_external_ssa_use_leaves_symbol_table_and_ir_unchanged() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "module".try_into().unwrap());
        let mut symbol_table = SymbolTable::new(&ctx, Box::new(module));
        let function_type = FunctionType::get(&ctx, vec![], vec![]);
        let function = FuncOp::new(&mut ctx, "used".try_into().unwrap(), function_type);
        symbol_table.insert(&ctx, Box::new(function), None).unwrap();
        Operation::push_result(function.get_operation(), &ctx, function_type.into());
        let result = function.get_operation().deref(&ctx).get_result(0);
        let user = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![result],
            vec![],
            0,
        );

        let function_ref = function.get_operation().deref(&ctx);
        let before = (
            ctx.operations.len(),
            ctx.regions.len(),
            ctx.basic_blocks.len(),
            ctx.value_counter.get(),
            ctx.use_counter.get(),
            ctx.ir_mutation_attempt_epoch().unwrap().value(),
            function_ref.get_parent_block(),
            function_ref.get_prev(),
            function_ref.get_next(),
            result.uses(&ctx),
            symbol_table.symbol_table.len(),
            symbol_table.symbol_table.capacity(),
        );
        drop(function_ref);

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                let _ = symbol_table.erase(&mut ctx, Box::new(function));
            }))
            .is_err()
        );
        let function_ref = function.get_operation().deref(&ctx);
        assert!(
            (
                ctx.operations.len(),
                ctx.regions.len(),
                ctx.basic_blocks.len(),
                ctx.value_counter.get(),
                ctx.use_counter.get(),
                ctx.ir_mutation_attempt_epoch().unwrap().value(),
                function_ref.get_parent_block(),
                function_ref.get_prev(),
                function_ref.get_next(),
                result.uses(&ctx),
                symbol_table.symbol_table.len(),
                symbol_table.symbol_table.capacity(),
            ) == before
        );
        assert!(
            symbol_table
                .lookup(&"used".try_into().unwrap())
                .is_some_and(|registered| registered.get_operation() == function.get_operation())
        );
        assert!(user.deref(&ctx).get_operand(0) == result);
    }

    #[test]
    fn remove_rejects_unregistered_same_name_operation_without_changes() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "module".try_into().unwrap());
        let mut symbol_table = SymbolTable::new(&ctx, Box::new(module));
        let function_type = FunctionType::get(&ctx, vec![], vec![]);
        let registered = FuncOp::new(&mut ctx, "duplicate".try_into().unwrap(), function_type);
        symbol_table
            .insert(&ctx, Box::new(registered), None)
            .unwrap();
        let unregistered = FuncOp::new(&mut ctx, "duplicate".try_into().unwrap(), function_type);
        unregistered
            .get_operation()
            .insert_at_back(module.get_body(&ctx, 0), &ctx);

        let registered_ref = registered.get_operation().deref(&ctx);
        let unregistered_ref = unregistered.get_operation().deref(&ctx);
        let before = (
            (
                ctx.operations.len(),
                ctx.regions.len(),
                ctx.basic_blocks.len(),
                ctx.value_counter.get(),
                ctx.use_counter.get(),
                ctx.ir_mutation_attempt_epoch().unwrap().value(),
            ),
            (
                registered_ref.get_parent_block(),
                registered_ref.get_prev(),
                registered_ref.get_next(),
                unregistered_ref.get_parent_block(),
                unregistered_ref.get_prev(),
                unregistered_ref.get_next(),
            ),
            (
                symbol_table.symbol_table.len(),
                symbol_table.symbol_table.capacity(),
            ),
        );
        drop(registered_ref);
        drop(unregistered_ref);

        assert!(symbol_table.remove(&ctx, Box::new(unregistered)).is_err());
        let registered_ref = registered.get_operation().deref(&ctx);
        let unregistered_ref = unregistered.get_operation().deref(&ctx);
        assert!(
            (
                (
                    ctx.operations.len(),
                    ctx.regions.len(),
                    ctx.basic_blocks.len(),
                    ctx.value_counter.get(),
                    ctx.use_counter.get(),
                    ctx.ir_mutation_attempt_epoch().unwrap().value(),
                ),
                (
                    registered_ref.get_parent_block(),
                    registered_ref.get_prev(),
                    registered_ref.get_next(),
                    unregistered_ref.get_parent_block(),
                    unregistered_ref.get_prev(),
                    unregistered_ref.get_next(),
                ),
                (
                    symbol_table.symbol_table.len(),
                    symbol_table.symbol_table.capacity(),
                ),
            ) == before
        );
        assert!(
            symbol_table
                .lookup(&"duplicate".try_into().unwrap())
                .is_some_and(|entry| entry.get_operation() == registered.get_operation())
        );
    }

    #[test]
    #[should_panic(expected = "Dangling Ptr deref")]
    fn test_symbol_table_erase() {
        let ctx = &mut Context::new();
        let module_op = ModuleOp::new(ctx, Identifier::try_from("module").unwrap());
        let mut symbol_table = SymbolTable::new(ctx, Box::new(module_op));

        let fty = FunctionType::get(ctx, vec![], vec![]);
        let f1 = FuncOp::new(ctx, "f1".try_into().unwrap(), fty);

        symbol_table.insert(ctx, Box::new(f1), None).unwrap();
        assert!(
            f1.get_operation().deref(ctx).get_parent_op(ctx) == Some(module_op.get_operation())
        );
        let f1_lookedup = symbol_table.lookup(&"f1".try_into().unwrap()).unwrap();
        assert!(f1_lookedup.get_operation() == f1.get_operation());

        symbol_table.erase(ctx, Box::new(f1)).unwrap();
        f1.get_operation().deref(ctx);
    }

    #[test]
    fn test_symbol_table_collection() -> Result<()> {
        let ctx = &mut Context::new();

        let module_op = ModuleOp::new(ctx, Identifier::try_from("module").unwrap());
        let nested_module_op = ModuleOp::new(ctx, Identifier::try_from("nested_module").unwrap());
        let mut symbol_table_collection = SymbolTableCollection::new();

        let fty = FunctionType::get(ctx, vec![], vec![]);
        let f1 = FuncOp::new(ctx, "f1".try_into().unwrap(), fty);

        symbol_table_collection
            .get_symbol_table(ctx, Box::new(module_op))
            .insert(ctx, Box::new(f1), None)?;

        let f1_lookedup = symbol_table_collection
            .lookup_symbol_in_table(ctx, Box::new(module_op), &"f1".try_into().unwrap())
            .unwrap();
        assert!(f1_lookedup.get_operation() == f1.get_operation());

        let f2 = FuncOp::new(ctx, "f2".try_into().unwrap(), fty);
        symbol_table_collection
            .get_symbol_table(ctx, Box::new(nested_module_op))
            .insert(ctx, Box::new(f2), None)?;

        let f2_lookedup = symbol_table_collection
            .lookup_symbol_in_table(ctx, Box::new(nested_module_op), &"f2".try_into().unwrap())
            .unwrap();

        assert!(f2_lookedup.get_operation() == f2.get_operation());

        symbol_table_collection
            .get_symbol_table(ctx, Box::new(module_op))
            .insert(ctx, Box::new(nested_module_op), None)?;

        let nested_module_lookedup = symbol_table_collection
            .lookup_symbol_in_nearest_table(
                ctx,
                f1.get_operation(),
                &"nested_module".try_into().unwrap(),
            )
            .unwrap();
        assert!(nested_module_lookedup.get_operation() == nested_module_op.get_operation());

        let f1_lookedup = symbol_table_collection
            .lookup_symbol_in_nearest_table(ctx, f1.get_operation(), &"f1".try_into().unwrap())
            .unwrap();
        assert!(f1_lookedup.get_operation() == f1.get_operation());

        Ok(())
    }
}
