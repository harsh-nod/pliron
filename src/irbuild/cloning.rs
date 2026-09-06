// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

//! Cloning IR entities with a value / block / op remapping.
//!
//! An [IrMapping] records, for every original IR entity, the clone that stands
//! in for it. Cloning an [Operation] rewrites each operand and successor through
//! the mapping; anything *not* in the mapping is left unchanged. This means that
//! uses of values defined outside the cloned scope will continue referring to the
//! originals.
//!
//! Cloning is **order-independent**, achieved conceptually in two passes.
//! 1. Clone and map (original -> cloned) definitions first. Definitions are
//!    op results, block arguments and basic blocks. The clones are just a "shell":
//!    they do not have any uses (operands / successors).
//! 2. Add mapped uses (operands / successors) to the cloned shell operations and
//!    clone nested regions.
//!
//! So a reference that points "forward" (a branch to a later block, or an operand
//! whose def isn't cloned yet) still resolves to its clone.
//!
//! New blocks and ops are inserted through a [Rewriter], so any listener it
//! carries is notified.

use alloc::vec::Vec;

use crate::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    irbuild::{
        inserter::{BlockInsertionPoint, Inserter, OpInsertionPoint},
        rewriter::{Rewriter, ScopedRewriter},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    operation::Operation,
    region::Region,
    result::Result,
    r#type::Typed,
    utils::table::IMap,
    value::Value,
};

/// Mapping between IR entities
#[derive(Debug, Default)]
pub struct IrMapping {
    values: IMap<Value, Value>,
    blocks: IMap<Ptr<BasicBlock>, Ptr<BasicBlock>>,
    ops: IMap<Ptr<Operation>, Ptr<Operation>>,
}

impl IrMapping {
    /// Create an empty mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Authenticate every mapping entry against `ctx` without changing the
    /// mapping or allocating temporary storage.
    fn try_preflight(&self, ctx: &Context) -> Result<()> {
        for (from, to) in &self.values {
            from.try_find_index(ctx)?;
            to.try_find_index(ctx)?;
        }
        for (from, to) in &self.blocks {
            drop(from.try_deref(ctx)?);
            drop(to.try_deref(ctx)?);
        }
        for (from, to) in &self.ops {
            drop(from.try_deref(ctx)?);
            drop(to.try_deref(ctx)?);
        }
        Ok(())
    }

    /// Record that `from` maps to `to`. Overwrites any existing entry.
    pub fn map_value(&mut self, from: Value, to: Value) {
        self.values.insert(from, to);
    }

    /// Record that `from` maps to `to`. Overwrites any existing entry.
    pub fn map_block(&mut self, from: Ptr<BasicBlock>, to: Ptr<BasicBlock>) {
        self.blocks.insert(from, to);
    }

    /// Record that `from` maps to `to`. Overwrites any existing entry.
    pub fn map_op(&mut self, from: Ptr<Operation>, to: Ptr<Operation>) {
        self.ops.insert(from, to);
    }

    /// Look up mapping for value `from`.
    pub fn lookup_value(&self, from: Value) -> Option<Value> {
        self.values.get(&from).copied()
    }

    /// Look up mapping for block `from`.
    pub fn lookup_block(&self, from: Ptr<BasicBlock>) -> Option<Ptr<BasicBlock>> {
        self.blocks.get(&from).copied()
    }

    /// Look up mapping for op `from`.
    pub fn lookup_op(&self, from: Ptr<Operation>) -> Option<Ptr<Operation>> {
        self.ops.get(&from).copied()
    }

    /// Look up mapping for value `from`. If none exists, returns `from`.
    pub fn lookup_value_or_default(&self, from: Value) -> Value {
        self.lookup_value(from).unwrap_or(from)
    }

    /// Look up mapping for block `from`. If none exists, returns `from`.
    pub fn lookup_block_or_default(&self, from: Ptr<BasicBlock>) -> Ptr<BasicBlock> {
        self.lookup_block(from).unwrap_or(from)
    }
}

/// Clone `op` (and the contents of its regions).
///
/// * Uses `mapper` for remapping operands and successors.
/// * Updates `mapper` with new clones.
///
/// The returned [Operation] is **unlinked** (i.e., not inserted in any block).
///
/// See module docs for algorithm details.
pub fn clone_operation(
    op: Ptr<Operation>,
    ctx: &mut Context,
    rewriter: &mut dyn Rewriter,
    mapper: &mut IrMapping,
) -> Ptr<Operation> {
    mapper
        .try_preflight(ctx)
        .expect("IR mapping preflight failed");
    Operation::try_preflight_tree(op, ctx).expect("Operation clone source preflight failed");

    let new_op = clone_op_shell(op, ctx, mapper);
    fill_operation(op, ctx, rewriter, mapper);
    new_op
}

/// Phase one of cloning an op:
/// 1. Build its clone with the right result types, successors and (empty) regions
///    but **no operands**
/// 2. Record the op and its results in `mapper`.
///
/// Note: Successors are already safe to remap: when cloning a block list, every clone
/// block is created and recorded before any op shell is built. This implementation detail
/// deviates from the conceptual algorithm described in the module doc.
///
/// The returned op is **unlinked**.
fn clone_op_shell(op: Ptr<Operation>, ctx: &mut Context, mapper: &mut IrMapping) -> Ptr<Operation> {
    let (concrete_op, result_types, successors, num_operands, num_regions, attributes, loc) = {
        let op_ref = op.deref(ctx);
        let successors: Vec<Ptr<BasicBlock>> = op_ref
            .successors()
            .map(|b| mapper.lookup_block_or_default(b))
            .collect();
        (
            op_ref.concrete_op_info(),
            op_ref.result_types().collect::<Vec<_>>(),
            successors,
            op_ref.get_num_operands(),
            op_ref.num_regions(),
            op_ref.attributes.clone(),
            op_ref.loc(),
        )
    };

    // No operands yet: they are pushed, remapped, in `fill_operation`.
    let new_op = Operation::new(
        ctx,
        concrete_op,
        result_types,
        Vec::with_capacity(num_operands),
        successors,
        num_regions,
    );
    {
        let mut new_ref = new_op.deref_mut(ctx);
        new_ref.attributes = attributes;
        new_ref.set_loc(loc);
    }

    // Record the op and its results for later use.
    mapper.map_op(op, new_op);
    let old_ref = op.deref(ctx);
    let new_ref = new_op.deref(ctx);
    for (old, new) in old_ref.results().zip(new_ref.results()) {
        mapper.map_value(old, new);
    }

    new_op
}

/// Phase two of cloning an op: add the clone's operands and clone the contents
/// of its nested regions.
fn fill_operation(
    op: Ptr<Operation>,
    ctx: &mut Context,
    rewriter: &mut dyn Rewriter,
    mapper: &mut IrMapping,
) {
    let new_op = mapper
        .lookup_op(op)
        .expect("op shell must be created before it is filled");

    // Operands, remapped through the now-complete mapping and pushed in order.
    let operands: Vec<Value> = op
        .deref(ctx)
        .operands()
        .map(|v| mapper.lookup_value_or_default(v))
        .collect();
    for operand in operands {
        Operation::push_operand(new_op, ctx, operand);
    }

    // Clone the blocks of each region into the corresponding (empty) region of
    // the clone.
    let num_regions = op.deref(ctx).num_regions();
    for region_idx in 0..num_regions {
        let src_region = op.deref(ctx).get_region(region_idx);
        let dest_region = new_op.deref(ctx).get_region(region_idx);
        clone_region_into(src_region, dest_region, ctx, rewriter, mapper);
    }
}

/// Clone every block of `src_region` into `dest_region`, appending at its end.
///
/// * Uses `mapper` for remapping operands and successors.
/// * Updates `mapper` with new clones.
///
/// See module docs for algorithm details.
pub fn clone_region_into(
    src_region: Ptr<Region>,
    dest_region: Ptr<Region>,
    ctx: &mut Context,
    rewriter: &mut dyn Rewriter,
    mapper: &mut IrMapping,
) {
    mapper
        .try_preflight(ctx)
        .expect("IR mapping preflight failed");
    Operation::try_preflight_region_tree(src_region, ctx)
        .expect("Region clone source preflight failed");
    Operation::try_preflight_region_tree(dest_region, ctx)
        .expect("Region clone destination preflight failed");

    let blocks: Vec<Ptr<BasicBlock>> = src_region.deref(ctx).iter(ctx).collect();
    clone_blocks_into(&blocks, dest_region, ctx, rewriter, mapper);
}

/// Clone `blocks` (and their operations) into `dest_region`, appending at its end
/// in the given order.
///
/// * Uses `mapper` for remapping operands and successors.
/// * Updates `mapper` with new clones.
///
/// See module docs for algorithm details.
pub fn clone_blocks_into(
    blocks: &[Ptr<BasicBlock>],
    dest_region: Ptr<Region>,
    ctx: &mut Context,
    rewriter: &mut dyn Rewriter,
    mapper: &mut IrMapping,
) {
    mapper
        .try_preflight(ctx)
        .expect("IR mapping preflight failed");
    Operation::try_preflight_region_tree(dest_region, ctx)
        .expect("Block clone destination preflight failed");
    for block in blocks {
        Operation::try_preflight_block_tree(*block, ctx)
            .expect("Block clone source preflight failed");
    }

    let mut rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);

    // Phase 1: create the clone blocks and their arguments, and record them.
    for &src_block in blocks {
        let (arg_types, label, attrs, loc) = {
            let block_ref = src_block.deref(ctx);
            let arg_types: Vec<_> = block_ref.arguments().map(|arg| arg.get_type(ctx)).collect();
            (
                arg_types,
                block_ref.label.clone(),
                block_ref.attributes.clone(),
                block_ref.loc(),
            )
        };
        let new_block = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AtRegionEnd(dest_region),
            label,
            arg_types,
        );

        {
            let mut new_ref = new_block.deref_mut(ctx);
            new_ref.attributes = attrs;
            new_ref.set_loc(loc);
        }

        let src_ref = src_block.deref(ctx);
        let new_ref = new_block.deref(ctx);
        for (old, new) in src_ref.arguments().zip(new_ref.arguments()) {
            mapper.map_value(old, new);
        }

        mapper.map_block(src_block, new_block);
    }

    // Phase 2: create each block's op shells (no operands, empty regions) and
    // record their results, so later phases can refer to them in any order.
    for &src_block in blocks {
        let new_block = mapper
            .lookup_block(src_block)
            .expect("block was mapped in phase one");
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(new_block));
        let ops: Vec<Ptr<Operation>> = src_block.deref(ctx).iter(ctx).collect();
        for src_op in ops {
            let shell = clone_op_shell(src_op, ctx, mapper);
            rewriter.append_operation(ctx, shell);
        }
    }

    // Phase 3: add operands and clone nested regions, now that every op result
    // in `blocks` is recorded.
    for &src_block in blocks {
        let ops: Vec<Ptr<Operation>> = src_block.deref(ctx).iter(ctx).collect();
        for src_op in ops {
            fill_operation(src_op, ctx, &mut rewriter, mapper);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builtin::{
            op_interfaces::{OneRegionInterface, SingleBlockRegionInterface},
            ops::ModuleOp,
            types::{IntegerType, Signedness},
        },
        irbuild::{listener::DummyListener, rewriter::IRRewriter},
        linked_list::ContainsLinkedList,
        op::Op,
    };
    use alloc::vec;
    use core::panic::AssertUnwindSafe;

    struct Fixture {
        module: ModuleOp,
        region: Ptr<Region>,
        block: Ptr<BasicBlock>,
        value_block: Ptr<BasicBlock>,
        value: Value,
    }

    #[derive(Eq, PartialEq)]
    struct ContextSnapshot {
        operation_count: usize,
        region_count: usize,
        block_count: usize,
        value_counter: u64,
        use_counter: u64,
        epoch: u64,
        module_regions: Vec<Ptr<Region>>,
        region_head: Option<Ptr<BasicBlock>>,
        region_tail: Option<Ptr<BasicBlock>>,
        block_head: Option<Ptr<Operation>>,
        block_tail: Option<Ptr<Operation>>,
        value_uses: Vec<crate::value::Use<Value>>,
        block_uses: Vec<crate::value::Use<Ptr<BasicBlock>>>,
    }

    fn fixture(ctx: &mut Context, name: &str) -> Fixture {
        let module = ModuleOp::new(ctx, name.try_into().unwrap());
        let region = module.get_region(ctx);
        let block = module.get_body(ctx, 0);
        let ty = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let value_block = BasicBlock::new(ctx, None, vec![ty]);
        let value = value_block.deref(ctx).get_argument(0);
        Fixture {
            module,
            region,
            block,
            value_block,
            value,
        }
    }

    fn snapshot(ctx: &Context, fixture: &Fixture) -> ContextSnapshot {
        let region_ref = fixture.region.deref(ctx);
        let region_head = region_ref.get_head();
        let region_tail = region_ref.get_tail();
        drop(region_ref);
        let block_ref = fixture.block.deref(ctx);
        let block_head = block_ref.get_head();
        let block_tail = block_ref.get_tail();
        drop(block_ref);
        ContextSnapshot {
            operation_count: ctx.operations.len(),
            region_count: ctx.regions.len(),
            block_count: ctx.basic_blocks.len(),
            value_counter: ctx.value_counter.get(),
            use_counter: ctx.use_counter.get(),
            epoch: ctx.ir_mutation_attempt_epoch().unwrap().value(),
            module_regions: fixture
                .module
                .get_operation()
                .deref(ctx)
                .regions()
                .collect(),
            region_head,
            region_tail,
            block_head,
            block_tail,
            value_uses: fixture.value.uses(ctx),
            block_uses: fixture.block.uses(ctx),
        }
    }

    fn assert_rejected_without_changes(
        owner: &mut Context,
        owner_fixture: &Fixture,
        foreign: &mut Context,
        foreign_fixture: &Fixture,
        mapper: &mut IrMapping,
        action: impl FnOnce(&mut Context, &mut IRRewriter<DummyListener>, &mut IrMapping),
    ) {
        let owner_before = snapshot(owner, owner_fixture);
        let foreign_before = snapshot(foreign, foreign_fixture);
        let mapper_before = (
            mapper.values.clone(),
            mapper.blocks.clone(),
            mapper.ops.clone(),
        );
        let mapper_capacities = (
            mapper.values.capacity(),
            mapper.blocks.capacity(),
            mapper.ops.capacity(),
        );
        let mut rewriter = IRRewriter::<DummyListener>::default();

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                action(owner, &mut rewriter, mapper)
            }))
            .is_err(),
            "foreign clone input unexpectedly succeeded"
        );
        assert!(snapshot(owner, owner_fixture) == owner_before);
        assert!(snapshot(foreign, foreign_fixture) == foreign_before);
        assert!(mapper.values == mapper_before.0);
        assert!(mapper.blocks == mapper_before.1);
        assert!(mapper.ops == mapper_before.2);
        assert_eq!(
            (
                mapper.values.capacity(),
                mapper.blocks.capacity(),
                mapper.ops.capacity(),
            ),
            mapper_capacities
        );
        assert!(!rewriter.is_modified());
        assert!(!rewriter.is_insertion_point_set());
    }

    #[test]
    fn foreign_equal_slot_clone_destinations_and_late_sources_leave_state_unchanged() {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = fixture(&mut owner, "owner");
        let foreign_fixture = fixture(&mut foreign, "foreign");
        assert_eq!(owner_fixture.region.idx, foreign_fixture.region.idx);
        assert_eq!(owner_fixture.block.idx, foreign_fixture.block.idx);
        assert_eq!(
            owner_fixture.value_block.idx,
            foreign_fixture.value_block.idx
        );

        assert_rejected_without_changes(
            &mut owner,
            &owner_fixture,
            &mut foreign,
            &foreign_fixture,
            &mut IrMapping::new(),
            |ctx, rewriter, mapper| {
                clone_region_into(
                    owner_fixture.region,
                    foreign_fixture.region,
                    ctx,
                    rewriter,
                    mapper,
                )
            },
        );

        assert_rejected_without_changes(
            &mut owner,
            &owner_fixture,
            &mut foreign,
            &foreign_fixture,
            &mut IrMapping::new(),
            |ctx, rewriter, mapper| {
                clone_blocks_into(
                    &[owner_fixture.block, foreign_fixture.block],
                    owner_fixture.region,
                    ctx,
                    rewriter,
                    mapper,
                )
            },
        );
    }

    enum MappingKind {
        Value,
        Block,
        Operation,
    }

    fn assert_foreign_mapping_rejected(kind: MappingKind) {
        let mut owner = Context::try_new().unwrap();
        let mut foreign = Context::try_new().unwrap();
        let owner_fixture = fixture(&mut owner, "owner");
        let foreign_fixture = fixture(&mut foreign, "foreign");
        let mut mapper = IrMapping::new();
        match kind {
            MappingKind::Value => {
                mapper.map_value(owner_fixture.value, foreign_fixture.value);
            }
            MappingKind::Block => {
                mapper.map_block(owner_fixture.block, foreign_fixture.block);
            }
            MappingKind::Operation => {
                mapper.map_op(
                    owner_fixture.module.get_operation(),
                    foreign_fixture.module.get_operation(),
                );
            }
        }

        assert_rejected_without_changes(
            &mut owner,
            &owner_fixture,
            &mut foreign,
            &foreign_fixture,
            &mut mapper,
            |ctx, rewriter, mapper| {
                clone_blocks_into(
                    &[owner_fixture.block],
                    owner_fixture.region,
                    ctx,
                    rewriter,
                    mapper,
                )
            },
        );
    }

    #[test]
    fn foreign_external_mapping_values_leave_clone_state_unchanged() {
        assert_foreign_mapping_rejected(MappingKind::Value);
    }

    #[test]
    fn foreign_external_mapping_blocks_leave_clone_state_unchanged() {
        assert_foreign_mapping_rejected(MappingKind::Block);
    }

    #[test]
    fn foreign_external_mapping_operations_leave_clone_state_unchanged() {
        assert_foreign_mapping_rejected(MappingKind::Operation);
    }

    #[test]
    fn tree_preflight_rejects_region_owned_by_another_operation_without_changes() {
        let mut ctx = Context::try_new().unwrap();
        let first = fixture(&mut ctx, "first");
        let second = fixture(&mut ctx, "second");
        first.module.get_operation().deref_mut(&ctx).regions[0] = second.region;
        let before = snapshot(&ctx, &first);

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                Operation::try_preflight_tree(first.module.get_operation(), &ctx).unwrap()
            }))
            .is_err()
        );
        assert!(snapshot(&ctx, &first) == before);
    }

    #[test]
    fn tree_preflight_rejects_duplicate_region_slots_without_changes() {
        let mut ctx = Context::try_new().unwrap();
        let op = Operation::new(
            &mut ctx,
            ModuleOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            2,
        );
        let regions: Vec<_> = op.deref(&ctx).regions().collect();
        op.deref_mut(&ctx).regions[1] = regions[0];
        let counts_before = (
            ctx.operations.len(),
            ctx.regions.len(),
            ctx.basic_blocks.len(),
            ctx.value_counter.get(),
            ctx.use_counter.get(),
            ctx.ir_mutation_attempt_epoch().unwrap().value(),
            op.deref(&ctx).regions().collect::<Vec<_>>(),
        );

        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                Operation::try_preflight_tree(op, &ctx).unwrap()
            }))
            .is_err()
        );
        assert_eq!(
            (
                ctx.operations.len(),
                ctx.regions.len(),
                ctx.basic_blocks.len(),
                ctx.value_counter.get(),
                ctx.use_counter.get(),
                ctx.ir_mutation_attempt_epoch().unwrap().value(),
                op.deref(&ctx).regions().collect::<Vec<_>>(),
            ),
            counts_before
        );
    }
}
