// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

use super::*;
use crate::{
    builtin::{
        op_interfaces::{IsTerminatorInterface, OneRegionInterface, SingleBlockRegionInterface},
        ops::{FuncOp, ModuleOp},
        types::{FunctionType, IntegerType, Signedness},
    },
    derive::pliron_op,
    op::Op,
    operation::{verify_operation, verify_value_dominance, verify_value_dominance_with_info},
};
use core::panic::AssertUnwindSafe;

#[pliron_op(name = "test.dominance_consume", format, verifier = "succ")]
struct ConsumeOp;

#[pliron_op(name = "test.dominance_produce", format, verifier = "succ")]
struct ProduceOp;

#[pliron_op(
    name = "test.dominance_branch",
    format,
    interfaces = [IsTerminatorInterface],
    verifier = "succ",
)]
struct BranchOp;

#[pliron_op(
    name = "test.dominance_scope",
    format,
    interfaces = [OneRegionInterface],
    verifier = "succ",
)]
struct ScopeOp;

fn branch(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    successors: Vec<Ptr<BasicBlock>>,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        BranchOp::get_concrete_op_info(),
        vec![],
        vec![],
        successors,
        0,
    );
    op.insert_at_back(block, ctx);
    op
}

fn consume(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        ConsumeOp::get_concrete_op_info(),
        vec![],
        vec![value],
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
    op
}

fn function(ctx: &mut Context, module: &ModuleOp, name: &str) -> FuncOp {
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let signature = FunctionType::get(ctx, vec![ty.into(), ty.into()], vec![]);
    let function = FuncOp::new(ctx, name.try_into().unwrap(), signature);
    module.append_operation(ctx, function.get_operation(), 0);
    function
}

fn fixture(ctx: &mut Context, count: usize) -> (ModuleOp, FuncOp, Vec<Ptr<Operation>>) {
    let module = ModuleOp::new(ctx, "dominance".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let block = function.get_entry_block(ctx);
    let arg = block.deref(ctx).get_argument(0);
    let users = (0..count).map(|_| consume(ctx, block, arg)).collect();
    branch(ctx, block, vec![]);
    (module, function, users)
}

fn must_panic(f: impl FnOnce()) {
    assert!(std::panic::catch_unwind(AssertUnwindSafe(f)).is_err());
}

fn outcome(f: impl FnOnce() -> bool) -> Option<bool> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).ok()
}

#[test]
fn one_scope_spans_complete_root_walk_and_preflights_once_per_used_region() {
    for count in [1, 32, 1024] {
        let ctx = &mut Context::new();
        let (module, first, _) = fixture(ctx, count);
        let second = function(ctx, &module, "g");
        let block = second.get_entry_block(ctx);
        let arg = block.deref(ctx).get_argument(0);
        for _ in 0..count {
            consume(ctx, block, arg);
        }
        branch(ctx, block, vec![]);
        verify_operation(module.get_operation(), ctx).unwrap();
        let epoch = ctx.ir_mutation_attempt_epoch().unwrap();
        let mut scoped = DomInfo::for_verification(ctx);
        verify_value_dominance_with_info(module.get_operation(), ctx, &mut scoped).unwrap();
        assert_eq!(scoped.verification_preflight_count(), 2);
        assert_eq!(scoped.0.len(), 2);
        assert!(scoped.0.contains_key(&first.get_region(ctx)));
        assert!(scoped.0.contains_key(&second.get_region(ctx)));

        let mut legacy = DomInfo::default();
        verify_value_dominance_with_info(module.get_operation(), ctx, &mut legacy).unwrap();
        assert_eq!(legacy.verification_preflight_count(), 2 * count);
        assert_eq!(legacy.0.len(), scoped.0.len());
        assert_eq!(ctx.ir_mutation_attempt_epoch().unwrap(), epoch);
    }
}

#[test]
fn cached_queries_keep_public_preflight_behavior_and_scoped_tree_identity() {
    let ctx = &mut Context::new();
    let (module, function, users) = fixture(ctx, 32);
    verify_operation(module.get_operation(), ctx).unwrap();
    let arg = function.get_entry_block(ctx).deref(ctx).get_argument(0);
    let mut legacy = DomInfo::default();
    let mut scoped = DomInfo::for_verification(ctx);
    for user in users {
        assert_eq!(
            legacy.value_strictly_dominates_op(ctx, arg, user),
            scoped.value_strictly_dominates_op(ctx, arg, user)
        );
    }
    assert_eq!(legacy.verification_preflight_count(), 32);
    assert_eq!(scoped.verification_preflight_count(), 1);
    let region = function.get_region(ctx);
    let original = scoped.get_dom_tree(ctx, region) as *const _;
    for _ in 0..32 {
        assert_eq!(scoped.get_dom_tree(ctx, region) as *const _, original);
    }
    assert_eq!(scoped.verification_preflight_count(), 1);
}

#[test]
fn diamond_loop_duplicate_edges_and_unreachable_queries_match_legacy() {
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "cfg".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let entry = function.get_entry_block(ctx);
    let region = function.get_region(ctx);
    let mut blocks = vec![entry];
    for _ in 0..5 {
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(region, ctx);
        blocks.push(block);
    }
    let arg = entry.deref(ctx).get_argument(0);
    let users: Vec<_> = blocks
        .iter()
        .map(|&block| consume(ctx, block, arg))
        .collect();
    branch(ctx, blocks[0], vec![blocks[1], blocks[2], blocks[2]]);
    branch(ctx, blocks[1], vec![blocks[3]]);
    branch(ctx, blocks[2], vec![blocks[3]]);
    branch(ctx, blocks[3], vec![blocks[3], blocks[4]]);
    branch(ctx, blocks[4], vec![]);
    branch(ctx, blocks[5], vec![blocks[5]]);
    let mut legacy = DomInfo::default();
    let mut scoped = DomInfo::for_verification(ctx);
    for &a in &blocks {
        for &b in &blocks {
            assert_eq!(
                outcome(|| legacy.block_strictly_dominates_block(ctx, a, b)),
                outcome(|| scoped.block_strictly_dominates_block(ctx, a, b))
            );
        }
    }
    for &user in &users {
        assert_eq!(
            outcome(|| legacy.value_strictly_dominates_op(ctx, arg, user)),
            outcome(|| scoped.value_strictly_dominates_op(ctx, arg, user))
        );
    }
    assert_eq!(scoped.verification_preflight_count(), 1);
    // Both cache modes must completely verify unreachable users of entry inputs.
    let mut old_walk = DomInfo::default();
    let mut new_walk = DomInfo::for_verification(ctx);
    assert_eq!(
        outcome(
            || verify_value_dominance_with_info(module.get_operation(), ctx, &mut old_walk).is_ok()
        ),
        Some(true)
    );
    assert_eq!(
        outcome(
            || verify_value_dominance_with_info(module.get_operation(), ctx, &mut new_walk).is_ok()
        ),
        Some(true)
    );
}

#[path = "dominance_unreachable_v1_tests.rs"]
mod unreachable_tests;

#[test]
fn nested_regions_do_not_borrow_each_others_preflight() {
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "nested".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let entry = function.get_entry_block(ctx);
    let outer_arg = entry.deref(ctx).get_argument(0);
    let scope = Operation::new(
        ctx,
        ScopeOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    scope.insert_at_back(entry, ctx);
    let inner_region = scope.deref(ctx).get_region(0);
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let inner = BasicBlock::new(ctx, None, vec![ty.into()]);
    inner.insert_at_back(inner_region, ctx);
    let inner_arg = inner.deref(ctx).get_argument(0);
    for _ in 0..8 {
        consume(ctx, inner, outer_arg);
        consume(ctx, inner, inner_arg);
    }
    branch(ctx, inner, vec![]);
    branch(ctx, entry, vec![]);
    verify_operation(module.get_operation(), ctx).unwrap();
    let mut info = DomInfo::for_verification(ctx);
    verify_value_dominance_with_info(module.get_operation(), ctx, &mut info).unwrap();
    assert_eq!(info.verification_preflight_count(), 2);
    assert!(info.0.contains_key(&function.get_region(ctx)));
    assert!(info.0.contains_key(&inner_region));
}

#[test]
fn failed_mutable_borrow_invalidates_a_warm_scope_without_changing_ir() {
    let ctx = &mut Context::new();
    let (module, function, users) = fixture(ctx, 1);
    let region = function.get_region(ctx);
    let mut info = DomInfo::for_verification(ctx);
    info.get_dom_tree(ctx, region);
    let before = ctx.ir_mutation_attempt_epoch().unwrap();
    let held = users[0].deref(ctx);
    assert!(users[0].try_deref_mut(ctx).is_err());
    assert!(ctx.ir_mutation_attempt_epoch().unwrap() > before);
    drop(held);
    must_panic(|| {
        info.get_dom_tree(ctx, region);
    });
    assert_eq!(info.verification_preflight_count(), 1);
    verify_operation(module.get_operation(), ctx).unwrap();
}

#[test]
fn operand_mutate_restore_and_same_block_queries_cannot_reuse_the_scope() {
    let ctx = &mut Context::new();
    let (module, function, users) = fixture(ctx, 1);
    let entry = function.get_entry_block(ctx);
    let first = entry.deref(ctx).get_argument(0);
    let second = entry.deref(ctx).get_argument(1);
    let mut info = DomInfo::for_verification(ctx);
    assert!(info.value_strictly_dominates_op(ctx, first, users[0]));
    Operation::replace_operand(users[0], ctx, 0, second);
    Operation::replace_operand(users[0], ctx, 0, first);
    must_panic(|| {
        info.value_strictly_dominates_op(ctx, first, users[0]);
    });
    assert_eq!(info.verification_preflight_count(), 1);
    verify_operation(module.get_operation(), ctx).unwrap();
}

#[test]
fn cfg_mutation_rejects_stale_tree_and_fresh_scope_observes_new_nondominance() {
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "mutation".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let entry = function.get_entry_block(ctx);
    let region = function.get_region(ctx);
    let definition = BasicBlock::new(ctx, None, vec![]);
    definition.insert_at_back(region, ctx);
    let use_block = BasicBlock::new(ctx, None, vec![]);
    use_block.insert_at_back(region, ctx);
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let producer = Operation::new(
        ctx,
        ProduceOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    producer.insert_at_back(definition, ctx);
    let value = producer.deref(ctx).get_result(0);
    let user = consume(ctx, use_block, value);
    let entry_branch = branch(ctx, entry, vec![definition]);
    branch(ctx, definition, vec![use_block]);
    branch(ctx, use_block, vec![]);
    verify_operation(module.get_operation(), ctx).unwrap();
    let mut info = DomInfo::for_verification(ctx);
    assert!(info.value_strictly_dominates_op(ctx, value, user));
    Operation::replace_successor(entry_branch, ctx, 0, use_block);
    must_panic(|| {
        info.value_strictly_dominates_op(ctx, value, user);
    });
    let mut fresh = DomInfo::for_verification(ctx);
    assert!(!fresh.value_strictly_dominates_op(ctx, value, user));
    assert!(verify_value_dominance(module.get_operation(), ctx).is_err());
    assert_eq!(info.verification_preflight_count(), 1);
}

#[test]
fn equal_slot_foreign_region_and_context_are_rejected_before_cache_changes() {
    let owner = &mut Context::new();
    let foreign = &mut Context::new();
    let (_, own_function, _) = fixture(owner, 1);
    let (_, foreign_function, _) = fixture(foreign, 1);
    let own_region = own_function.get_region(owner);
    let foreign_region = foreign_function.get_region(foreign);
    assert_eq!(own_region.idx, foreign_region.idx);
    let epochs = (
        owner.ir_mutation_attempt_epoch().unwrap(),
        foreign.ir_mutation_attempt_epoch().unwrap(),
    );
    let mut info = DomInfo::for_verification(owner);
    info.get_dom_tree(owner, own_region);
    let before = (
        info.0.len(),
        info.0.capacity(),
        info.verification_preflight_count(),
    );
    must_panic(|| {
        info.get_dom_tree(owner, foreign_region);
    });
    must_panic(|| {
        info.get_dom_tree(foreign, foreign_region);
    });
    assert_eq!(
        (
            info.0.len(),
            info.0.capacity(),
            info.verification_preflight_count()
        ),
        before
    );
    assert_eq!(
        (
            owner.ir_mutation_attempt_epoch().unwrap(),
            foreign.ir_mutation_attempt_epoch().unwrap()
        ),
        epochs
    );
    info.get_dom_tree(owner, own_region);
}

#[test]
fn erased_region_does_not_reuse_a_cached_tree() {
    let ctx = &mut Context::new();
    let (_, function, _) = fixture(ctx, 1);
    let region = function.get_region(ctx);
    let mut info = DomInfo::for_verification(ctx);
    info.get_dom_tree(ctx, region);
    Operation::erase(function.get_operation(), ctx);
    assert!(region.try_deref(ctx).is_err());
    must_panic(|| {
        info.get_dom_tree(ctx, region);
    });
    let mut fresh = DomInfo::for_verification(ctx);
    must_panic(|| {
        fresh.get_dom_tree(ctx, region);
    });
    assert_eq!(fresh.verification_preflight_count(), 0);
    assert!(fresh.0.is_empty());
}

#[test]
fn same_block_result_use_before_definition_keeps_original_error_without_tree() {
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "bad_order".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let block = function.get_entry_block(ctx);
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let producer = Operation::new(
        ctx,
        ProduceOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    let value = producer.deref(ctx).get_result(0);
    let user = consume(ctx, block, value);
    producer.insert_at_back(block, ctx);
    branch(ctx, block, vec![]);
    let mut legacy = DomInfo::default();
    let old_error =
        verify_value_dominance_with_info(module.get_operation(), ctx, &mut legacy).unwrap_err();
    let mut scoped = DomInfo::for_verification(ctx);
    let new_error =
        verify_value_dominance_with_info(module.get_operation(), ctx, &mut scoped).unwrap_err();
    assert_eq!(
        alloc::format!("{old_error:?}"),
        alloc::format!("{new_error:?}")
    );
    assert_eq!(scoped.verification_preflight_count(), 0);
    assert!(!scoped.value_strictly_dominates_op(ctx, value, user));
    block
        .deref_mut(ctx)
        .set_label(Some("renamed".try_into().unwrap()));
    must_panic(|| {
        scoped.value_strictly_dominates_op(ctx, value, user);
    });
}
