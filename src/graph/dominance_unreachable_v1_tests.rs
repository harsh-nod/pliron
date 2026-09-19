// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

//! Complete verifier controls, not an unreachable-block verification exemption.
use super::*;
use crate::builtin::op_interfaces::{IsolatedFromAboveInterface, SameOperandsType};
use alloc::string::ToString;

#[pliron_op(name = "test.unreachable_same_type", format, interfaces = [SameOperandsType], verifier = "succ")]
struct SameTypeOp;

#[pliron_op(name = "test.unreachable_isolated", format, interfaces = [OneRegionInterface, IsolatedFromAboveInterface], verifier = "succ")]
struct IsolatedScopeOp;

fn block(ctx: &mut Context, function: &FuncOp) -> Ptr<BasicBlock> {
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(function.get_region(ctx), ctx);
    block
}

fn produce(ctx: &mut Context, block: Ptr<BasicBlock>) -> Ptr<Operation> {
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let op = Operation::new(
        ctx,
        ProduceOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
    op
}

fn check_both(ctx: &Context, module: &ModuleOp, expected: bool) {
    let mut legacy = DomInfo::default();
    let mut scoped = DomInfo::for_verification(ctx);
    assert_eq!(
        verify_value_dominance_with_info(module.get_operation(), ctx, &mut legacy).is_ok(),
        expected
    );
    assert_eq!(
        verify_value_dominance_with_info(module.get_operation(), ctx, &mut scoped).is_ok(),
        expected
    );
}

#[test]
fn entry_inputs_are_available_without_adding_unreachable_dominance_edges() {
    for topology in 0..4 {
        let ctx = &mut Context::new();
        let module = ModuleOp::new(ctx, "entry_inputs".try_into().unwrap());
        let function = function(ctx, &module, "f");
        let entry = function.get_entry_block(ctx);
        let dead = block(ctx, &function);
        let other = block(ctx, &function);
        let arg = entry.deref(ctx).get_argument(0);
        let dead_user = consume(ctx, dead, arg);
        let other_user = consume(ctx, other, arg);
        branch(ctx, entry, vec![]);
        branch(
            ctx,
            dead,
            match topology {
                0 => vec![],
                1 => vec![dead],
                2 => vec![entry],
                _ => vec![other, other],
            },
        );
        branch(ctx, other, vec![]);
        let epoch = ctx.ir_mutation_attempt_epoch().unwrap();
        verify_operation(module.get_operation(), ctx).unwrap();
        check_both(ctx, &module, true);
        let mut info = DomInfo::for_verification(ctx);
        assert!(info.value_strictly_dominates_op(ctx, arg, dead_user));
        assert!(info.value_strictly_dominates_op(ctx, arg, other_user));
        assert!(!info.block_strictly_dominates_block(ctx, entry, dead));
        assert!(!info.block_strictly_dominates_block(ctx, dead, other));
        assert_eq!(info.nearest_common_dominator(ctx, entry, dead), None);
        assert_eq!(info.nearest_common_dominator(ctx, dead, other), None);
        assert_eq!(info.nearest_common_dominator(ctx, dead, dead), Some(dead));
        let tree = info.get_dom_tree(ctx, function.get_region(ctx));
        assert_eq!(tree.num_nodes(), 1);
        assert!(tree.contains(&entry));
        assert!(!tree.contains(&dead));
        for a in [entry, dead, other] {
            for b in [entry, dead, other] {
                assert_eq!(tree.dominates(&a, &b), a == b);
            }
        }
        assert_eq!(ctx.ir_mutation_attempt_epoch().unwrap(), epoch);
    }
}

#[test]
fn unreachable_operation_results_do_not_gain_cross_block_availability() {
    for defining_block_is_entry in [false, true] {
        let ctx = &mut Context::new();
        let module = ModuleOp::new(ctx, "operation_results".try_into().unwrap());
        let function = function(ctx, &module, "f");
        let entry = function.get_entry_block(ctx);
        let dead = block(ctx, &function);
        let other = block(ctx, &function);
        let definition = produce(ctx, if defining_block_is_entry { entry } else { dead });
        let value = definition.deref(ctx).get_result(0);
        let user = consume(ctx, other, value);
        for block in [entry, dead, other] {
            branch(ctx, block, vec![]);
        }
        let mut info = DomInfo::for_verification(ctx);
        assert!(!info.value_strictly_dominates_op(ctx, value, user));
        assert!(!info.op_strictly_dominates_op(ctx, definition, user));
        assert!(verify_operation(module.get_operation(), ctx).is_err());
        check_both(ctx, &module, false);
    }
}

#[test]
fn unreachable_same_block_order_and_nonentry_argument_scope_remain_strict() {
    for forward in [false, true] {
        let ctx = &mut Context::new();
        let module = ModuleOp::new(ctx, "local_order".try_into().unwrap());
        let function = function(ctx, &module, "f");
        let entry = function.get_entry_block(ctx);
        let dead = block(ctx, &function);
        let ty = IntegerType::get(ctx, 32, Signedness::Signed);
        let definition = Operation::new(
            ctx,
            ProduceOp::get_concrete_op_info(),
            vec![ty.into()],
            vec![],
            vec![],
            0,
        );
        let value = definition.deref(ctx).get_result(0);
        if !forward {
            definition.insert_at_back(dead, ctx);
        }
        consume(ctx, dead, value);
        if forward {
            definition.insert_at_back(dead, ctx);
        }
        branch(ctx, entry, vec![]);
        branch(ctx, dead, vec![]);
        assert_eq!(
            verify_operation(module.get_operation(), ctx).is_ok(),
            !forward
        );
        check_both(ctx, &module, !forward);
    }
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "nonentry_argument".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let entry = function.get_entry_block(ctx);
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let dead = BasicBlock::new(ctx, None, vec![ty.into()]);
    dead.insert_at_back(function.get_region(ctx), ctx);
    let other = block(ctx, &function);
    let value = dead.deref(ctx).get_argument(0);
    let local = consume(ctx, dead, value);
    let escaped = consume(ctx, other, value);
    let reachable = consume(ctx, entry, value);
    for block in [entry, dead, other] {
        branch(ctx, block, vec![]);
    }
    let mut info = DomInfo::for_verification(ctx);
    assert!(info.value_strictly_dominates_op(ctx, value, local));
    assert!(!info.value_strictly_dominates_op(ctx, value, escaped));
    assert!(!info.value_strictly_dominates_op(ctx, value, reachable));
    check_both(ctx, &module, false);
}

#[test]
fn reachable_sibling_non_dominance_still_rejects() {
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "reachable_siblings".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let entry = function.get_entry_block(ctx);
    let left = block(ctx, &function);
    let right = block(ctx, &function);
    let def = produce(ctx, left);
    let value = def.deref(ctx).get_result(0);
    let user = consume(ctx, right, value);
    branch(ctx, entry, vec![left, right]);
    branch(ctx, left, vec![]);
    branch(ctx, right, vec![]);
    let mut info = DomInfo::for_verification(ctx);
    assert!(!info.value_strictly_dominates_op(ctx, value, user));
    assert_eq!(info.nearest_common_dominator(ctx, left, right), Some(entry));
    assert!(verify_operation(module.get_operation(), ctx).is_err());
    check_both(ctx, &module, false);
}

#[test]
fn unreachable_operation_type_verification_is_not_skipped() {
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "unreachable_types".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let entry = function.get_entry_block(ctx);
    let ty = IntegerType::get(ctx, 64, Signedness::Signed);
    let dead = BasicBlock::new(ctx, None, vec![ty.into()]);
    dead.insert_at_back(function.get_region(ctx), ctx);
    let i32_arg = entry.deref(ctx).get_argument(0);
    let i64_arg = dead.deref(ctx).get_argument(0);
    let op = Operation::new(
        ctx,
        SameTypeOp::get_concrete_op_info(),
        vec![],
        vec![i32_arg, i64_arg],
        vec![],
        0,
    );
    op.insert_at_back(dead, ctx);
    branch(ctx, entry, vec![]);
    branch(ctx, dead, vec![]);
    check_both(ctx, &module, true);
    let error = verify_operation(module.get_operation(), ctx).unwrap_err();
    assert!(error.to_string().contains("different operand types"));
}

fn scope(
    ctx: &mut Context,
    parent: Ptr<BasicBlock>,
    isolated: bool,
) -> (Ptr<Region>, Ptr<BasicBlock>) {
    let op = Operation::new(
        ctx,
        if isolated {
            IsolatedScopeOp::get_concrete_op_info()
        } else {
            ScopeOp::get_concrete_op_info()
        },
        vec![],
        vec![],
        vec![],
        1,
    );
    op.insert_at_back(parent, ctx);
    let region = op.deref(ctx).get_region(0);
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let entry = BasicBlock::new(ctx, None, vec![ty.into()]);
    entry.insert_at_back(region, ctx);
    (region, entry)
}

#[test]
fn new_unreachable_lexical_availability_does_not_cross_isolated_namespaces() {
    for isolated in [false, true] {
        for unreachable in [false, true] {
            let ctx = &mut Context::new();
            let module = ModuleOp::new(ctx, "nested_unreachable".try_into().unwrap());
            let function = function(ctx, &module, "f");
            let entry = function.get_entry_block(ctx);
            let dead = block(ctx, &function);
            let value = entry.deref(ctx).get_argument(0);
            let (_, inner) = scope(ctx, if unreachable { dead } else { entry }, isolated);
            let user = consume(ctx, inner, value);
            for block in [entry, dead, inner] {
                branch(ctx, block, vec![]);
            }
            let mut info = DomInfo::for_verification(ctx);
            // Existing reachable queries do not enforce this namespace interface as
            // a capture barrier. This patch does not claim to repair that contract.
            let allowed = !unreachable || !isolated;
            assert_eq!(info.value_strictly_dominates_op(ctx, value, user), allowed);
            assert_eq!(
                verify_operation(module.get_operation(), ctx).is_ok(),
                allowed
            );
            check_both(ctx, &module, allowed);
        }
    }
}

#[test]
fn nested_inputs_do_not_escape_into_parent_or_sibling_regions() {
    for escape_parent in [false, true] {
        let ctx = &mut Context::new();
        let module = ModuleOp::new(ctx, "lexical_escape".try_into().unwrap());
        let function = function(ctx, &module, "f");
        let entry = function.get_entry_block(ctx);
        let dead = block(ctx, &function);
        let (_, inner) = scope(ctx, dead, false);
        let (_, sibling) = scope(ctx, dead, false);
        let value = inner.deref(ctx).get_argument(0);
        let local = consume(ctx, inner, value);
        let user = consume(ctx, if escape_parent { dead } else { sibling }, value);
        for block in [entry, dead, inner, sibling] {
            branch(ctx, block, vec![]);
        }
        let mut info = DomInfo::for_verification(ctx);
        assert!(info.value_strictly_dominates_op(ctx, value, local));
        assert!(!info.value_strictly_dominates_op(ctx, value, user));
        check_both(ctx, &module, false);
    }
}

#[test]
fn entry_input_extension_keeps_foreign_context_and_stale_epoch_checks() {
    let ctx = &mut Context::new();
    let module = ModuleOp::new(ctx, "provenance".try_into().unwrap());
    let function = function(ctx, &module, "f");
    let entry = function.get_entry_block(ctx);
    let dead = block(ctx, &function);
    let value = entry.deref(ctx).get_argument(0);
    let user = consume(ctx, dead, value);
    branch(ctx, entry, vec![]);
    branch(ctx, dead, vec![]);
    let mut scoped = DomInfo::for_verification(ctx);
    assert!(scoped.value_strictly_dominates_op(ctx, value, user));
    let foreign = &mut Context::new();
    must_panic(|| {
        DomInfo::default().value_strictly_dominates_op(foreign, value, user);
    });
    let held = user.deref(ctx);
    assert!(user.try_deref_mut(ctx).is_err());
    drop(held);
    must_panic(|| {
        scoped.value_strictly_dominates_op(ctx, value, user);
    });
    let mut fresh = DomInfo::for_verification(ctx);
    assert!(fresh.value_strictly_dominates_op(ctx, value, user));
    Operation::erase(user, ctx);
    assert!(user.try_deref(ctx).is_err());
    must_panic(|| {
        fresh.value_strictly_dominates_op(ctx, value, user);
    });
    must_panic(|| {
        DomInfo::default().value_strictly_dominates_op(ctx, value, user);
    });
}
