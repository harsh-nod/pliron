// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

mod common;

use std::sync::{Arc, Mutex};

#[cfg(target_family = "wasm")]
use wasm_bindgen_test::wasm_bindgen_test;

use common::{ConstantOp, ReturnOp, const_ret_in_mod};
use pliron::{
    builtin::op_interfaces::OneResultInterface,
    context::{Context, Ptr},
    irbuild::{
        inserter::Inserter,
        listener::{Recorder, RecorderEvent},
        match_rewrite::{
            MatchRewrite, MatchRewriter, RewriterOrder, apply_match_rewrite,
            apply_match_rewrite_with_observer,
        },
        observer::{RewriteEvent, RewriteObserver},
        rewriter::{IRRewriter, Rewriter},
    },
    op::{Op, verify_op},
    operation::Operation,
    printable::Printable,
    result::Result,
};

#[cfg(not(target_family = "wasm"))]
use pliron::{
    irbuild::IRStatus,
    opts::{
        constants::sccp::{sccp, sccp_with_observer},
        dce::{dce, dce_with_observer},
        simplify_cfg::{simplify_cfg, simplify_cfg_with_observer},
    },
};

#[derive(Debug, Eq, PartialEq)]
enum Seen {
    Insert,
    Erase,
    BlockErase,
    ValueReplacement,
    OperationReplacement { results: usize },
}

struct Observer(Arc<Mutex<Vec<Seen>>>);

impl RewriteObserver for Observer {
    fn observe(&mut self, ctx: &Context, event: RewriteEvent) {
        let seen = match event {
            RewriteEvent::OperationInserted(op) => {
                assert!(op.try_deref(ctx).is_ok());
                Some(Seen::Insert)
            }
            RewriteEvent::OperationErased(op) => {
                assert!(op.try_deref(ctx).is_ok());
                Some(Seen::Erase)
            }
            RewriteEvent::BlockErased(block) => {
                assert!(block.try_deref(ctx).is_ok());
                Some(Seen::BlockErase)
            }
            RewriteEvent::ValueReplaced { old, new } => {
                assert!(old.try_find_index(ctx).is_ok());
                assert!(new.try_find_index(ctx).is_ok());
                Some(Seen::ValueReplacement)
            }
            RewriteEvent::OperationReplaced { old, new } => {
                assert!(new.try_deref(ctx).is_ok());
                Some(Seen::OperationReplacement {
                    results: old.deref(ctx).get_num_results(),
                })
            }
            _ => None,
        };
        if let Some(seen) = seen {
            self.0.lock().unwrap().push(seen);
        }
    }
}

fn observer(seen: &Arc<Mutex<Vec<Seen>>>) -> Box<dyn RewriteObserver> {
    Box::new(Observer(Arc::clone(seen)))
}

#[cfg_attr(not(target_family = "wasm"), test)]
#[cfg_attr(target_family = "wasm", wasm_bindgen_test)]
fn recorder_struct_literals_remain_source_compatible() {
    let recorder = Recorder { events: Vec::new() };
    assert!(recorder.events.is_empty());
}

#[cfg_attr(not(target_family = "wasm"), test)]
#[cfg_attr(target_family = "wasm", wasm_bindgen_test)]
fn observes_zero_result_replacement_before_erasure_without_changing_legacy_events() -> Result<()> {
    let ctx = &mut Context::new();
    let (module, _, constant, old_return) = const_ret_in_mod(ctx)?;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.set_observer(Some(observer(&seen)));
    rewriter.set_insertion_point_before_operation(old_return.get_operation());
    let replacement = ReturnOp::new(ctx, constant.get_result(ctx));
    rewriter.insert_operation(ctx, replacement.get_operation());
    rewriter.replace_operation(ctx, old_return.get_operation(), replacement.get_operation());
    verify_op(&module, ctx)?;
    assert_eq!(
        *seen.lock().unwrap(),
        [
            Seen::Insert,
            Seen::OperationReplacement { results: 0 },
            Seen::Erase,
        ]
    );
    assert_eq!(rewriter.get_listener().events.len(), 2);
    assert!(matches!(
        rewriter.get_listener().events.as_slice(),
        [
            RecorderEvent::InsertedOperation(_),
            RecorderEvent::ErasedOperation(_)
        ]
    ));
    Ok(())
}

// The LLVM dialect is already used by the pinned optimizer integration tests
// and is not supported on wasm. The basic rewriter tests above remain portable.
#[cfg(not(target_family = "wasm"))]
type ObservedPass = fn(Ptr<Operation>, &mut Context, Box<dyn RewriteObserver>) -> Result<IRStatus>;

#[cfg(not(target_family = "wasm"))]
fn run_changing_pass_pair(
    input: &str,
    plain: fn(Ptr<Operation>, &mut Context) -> Result<IRStatus>,
    observed: ObservedPass,
) -> Result<Vec<Seen>> {
    use pliron::{
        irfmt::parsers::spaced, operation::verify_operation, parsable::parse_from_str,
        result::ExpectOk,
    };
    use pliron_llvm as _;

    let a = &mut Context::new();
    let root_a = parse_from_str(spaced(Operation::top_level_parser()), a, input).expect_ok(a);
    let b = &mut Context::new();
    let root_b = parse_from_str(spaced(Operation::top_level_parser()), b, input).expect_ok(b);
    verify_operation(root_a, a)?;
    verify_operation(root_b, b)?;
    let before = Operation::get_op_dyn(root_a, a).disp(a).to_string();
    let seen = Arc::new(Mutex::new(Vec::new()));
    assert_eq!(plain(root_a, a)?, IRStatus::Changed);
    assert_eq!(observed(root_b, b, observer(&seen))?, IRStatus::Changed);
    let after_a = Operation::get_op_dyn(root_a, a).disp(a).to_string();
    let after_b = Operation::get_op_dyn(root_b, b).disp(b).to_string();
    assert_ne!(before, after_a);
    assert_eq!(after_a, after_b);
    verify_operation(root_a, a)?;
    verify_operation(root_b, b)?;
    let events = std::mem::take(&mut *seen.lock().unwrap());
    assert!(!events.is_empty());
    Ok(events)
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn observed_sccp_matches_plain_materialization_and_value_replacement() -> Result<()> {
    let input = r#"
    llvm.func @f: llvm.func <builtin.integer i64 () variadic = false> [] {
      ^entry():
      a = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      b = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
      sum = llvm.add a, b <{nsw=false,nuw=false}> : builtin.integer i64;
      llvm.return sum
    }
    "#;
    let events = run_changing_pass_pair(input, sccp, sccp_with_observer)?;
    assert!(events.contains(&Seen::Insert));
    assert!(events.contains(&Seen::ValueReplacement));
    Ok(())
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn observed_dce_notifications_survive_each_legacy_recorder_drain() -> Result<()> {
    let input = r#"
    llvm.func @f: llvm.func <builtin.integer i64 () variadic = false> [] {
      ^entry():
      live = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
      dead_a = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
      dead_b = llvm.add dead_a, dead_a <{nsw=false,nuw=false}> : builtin.integer i64;
      llvm.return live
    }
    "#;
    let events = run_changing_pass_pair(input, dce, dce_with_observer)?;
    assert_eq!(events, [Seen::Erase, Seen::Erase]);
    Ok(())
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn observed_cfg_merge_matches_plain_value_replacement_and_erasure() -> Result<()> {
    let input = r#"
    llvm.func @f: llvm.func <builtin.integer i64 () variadic = false> [] {
      ^entry():
      c = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
      llvm.br ^bb1(c)

      ^bb1(x: builtin.integer i64):
      llvm.return x
    }
    "#;
    let events = run_changing_pass_pair(input, simplify_cfg, simplify_cfg_with_observer)?;
    assert!(events.contains(&Seen::ValueReplacement));
    assert!(events.contains(&Seen::Erase));
    assert!(events.contains(&Seen::BlockErase));
    Ok(())
}

#[derive(Default)]
struct ReplaceReturn {
    done: bool,
}
impl MatchRewrite for ReplaceReturn {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        !self.done && Operation::is_op::<ReturnOp>(op, ctx)
    }
    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let old = Operation::get_op::<ReturnOp>(op, ctx).unwrap();
        let value = old.get_operation().deref(ctx).get_operand(0);
        let replacement = ReturnOp::new(ctx, value).get_operation();
        rewriter.insert_operation(ctx, replacement);
        rewriter.replace_operation(ctx, op, replacement);
        self.done = true;
        Ok(())
    }
}

#[cfg_attr(not(target_family = "wasm"), test)]
#[cfg_attr(target_family = "wasm", wasm_bindgen_test)]
fn opt_in_match_driver_retains_default_rewrite_scheduling_and_output() -> Result<()> {
    let a = &mut Context::new();
    let (root_a, _, _, _) = const_ret_in_mod(a)?;
    let b = &mut Context::new();
    let (root_b, _, _, _) = const_ret_in_mod(b)?;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let plain = apply_match_rewrite(
        a,
        &mut ReplaceReturn::default(),
        RewriterOrder::default(),
        root_a.get_operation(),
    )?;
    let observed = apply_match_rewrite_with_observer(
        b,
        &mut ReplaceReturn::default(),
        RewriterOrder::default(),
        root_b.get_operation(),
        observer(&seen),
    )?;
    assert_eq!(plain, observed);
    assert_eq!(root_a.disp(a).to_string(), root_b.disp(b).to_string());
    assert!(
        seen.lock()
            .unwrap()
            .contains(&Seen::OperationReplacement { results: 0 })
    );
    Ok(())
}

#[cfg_attr(not(target_family = "wasm"), test)]
#[cfg_attr(target_family = "wasm", wasm_bindgen_test)]
fn default_rewriter_retains_positive_auto_traits() {
    fn require<T: Send + Sync>() {}
    fn require_observer<T: std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
    // Recorder contains context-owned pointer/value descriptors, not Context.
    // This is a compile-time contract check, not permission to share Context.
    require::<IRRewriter<Recorder>>();
    // The existing rewriter contains dynamic attributes and is not unwind-safe.
    require_observer::<Box<dyn RewriteObserver>>();
}

#[cfg_attr(not(target_family = "wasm"), test)]
#[cfg_attr(target_family = "wasm", wasm_bindgen_test)]
fn arena_reuse_does_not_equal_an_erased_operation_identity() -> Result<()> {
    let ctx = &mut Context::new();
    let (root, _, _, ret) = const_ret_in_mod(ctx)?;
    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.set_insertion_point_before_operation(ret.get_operation());
    let old = ConstantOp::new(ctx, 41).get_operation();
    rewriter.insert_operation(ctx, old);
    rewriter.erase_operation(ctx, old);
    for value in 0..64 {
        let new = ConstantOp::new(ctx, value).get_operation();
        assert_ne!(old, new);
        rewriter.insert_operation(ctx, new);
        rewriter.erase_operation(ctx, new);
    }
    verify_op(&root, ctx)?;
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[derive(Debug, PartialEq, Eq)]
enum OccurrenceSeen {
    DropUses {
        operations: usize,
        successors: usize,
    },
    Argument {
        index: usize,
        incoming_occurrences: Vec<usize>,
        unequal_live_inputs: bool,
    },
    Merge {
        actual_count: usize,
        formal_count: usize,
    },
}

#[cfg(not(target_family = "wasm"))]
struct OccurrenceObserver {
    enabled: bool,
    legacy: Observer,
    seen: Arc<Mutex<Vec<OccurrenceSeen>>>,
}

#[cfg(not(target_family = "wasm"))]
impl RewriteObserver for OccurrenceObserver {
    fn observe(&mut self, ctx: &Context, event: RewriteEvent) {
        self.legacy.observe(ctx, event);
    }

    fn observes_occurrences(&self) -> bool {
        self.enabled
    }

    fn observe_occurrence(
        &mut self,
        ctx: &Context,
        event: pliron::irbuild::observer::RewriteOccurrenceEvent,
    ) {
        use pliron::{
            builtin::op_interfaces::BranchOpInterface, irbuild::observer::RewriteOccurrenceEvent,
            op::op_cast,
        };
        assert!(
            self.enabled,
            "disabled observers must not receive discarded events"
        );
        let seen = match event {
            RewriteOccurrenceEvent::UnreachableBlockDroppingUses { block } => {
                use pliron::linked_list::ContainsLinkedList;
                let operations = block.deref(ctx).iter(ctx).count();
                let terminator = block.deref(ctx).get_terminator(ctx).unwrap();
                OccurrenceSeen::DropUses {
                    operations,
                    successors: terminator.deref(ctx).get_num_successors(),
                }
            }
            RewriteOccurrenceEvent::BlockArgumentErasing { block, argument } => {
                let index = argument.find_index(ctx);
                assert_eq!(block.deref(ctx).get_argument(index), argument);
                let mut occurrences = Vec::new();
                let mut live_inputs = Vec::new();
                for predecessor in block.uses(ctx) {
                    let ordinal = predecessor.find_index(ctx);
                    let raw = Operation::get_op_dyn(predecessor.user_op(), ctx);
                    let branch = op_cast::<dyn BranchOpInterface>(&*raw).unwrap();
                    let inputs = branch.successor_operands(ctx, ordinal);
                    assert_eq!(inputs.len(), block.deref(ctx).get_num_arguments());
                    occurrences.push(ordinal);
                    live_inputs.push(inputs[0]);
                }
                occurrences.sort_unstable();
                OccurrenceSeen::Argument {
                    index,
                    incoming_occurrences: occurrences,
                    unequal_live_inputs: live_inputs.len() == 2 && live_inputs[0] != live_inputs[1],
                }
            }
            RewriteOccurrenceEvent::SuccessorBlockMerging {
                predecessor,
                successor,
                terminator,
            } => {
                assert_eq!(predecessor.deref(ctx).get_terminator(ctx), Some(terminator));
                assert_eq!(terminator.deref(ctx).get_num_successors(), 1);
                assert_eq!(terminator.deref(ctx).get_successor(0), successor);
                assert_eq!(successor.num_preds(ctx), 1);
                let raw = Operation::get_op_dyn(terminator, ctx);
                let branch = op_cast::<dyn BranchOpInterface>(&*raw).unwrap();
                OccurrenceSeen::Merge {
                    actual_count: branch.successor_operands(ctx, 0).len(),
                    formal_count: successor.deref(ctx).get_num_arguments(),
                }
            }
            RewriteOccurrenceEvent::BranchSuccessorSelected { .. } => {
                panic!("the generic merge/DCE fixtures do not select conditional branches")
            }
        };
        self.seen.lock().unwrap().push(seen);
    }
}

#[cfg(not(target_family = "wasm"))]
fn run_occurrence_pass(
    input: &str,
    pass: ObservedPass,
    enabled: bool,
) -> Result<(String, Vec<Seen>, Vec<OccurrenceSeen>)> {
    use pliron::{
        irfmt::parsers::spaced, operation::verify_operation, parsable::parse_from_str,
        result::ExpectOk,
    };
    use pliron_llvm as _;
    let ctx = &mut Context::new();
    let root = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(root, ctx)?;
    let legacy = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    assert_eq!(
        pass(
            root,
            ctx,
            Box::new(OccurrenceObserver {
                enabled,
                legacy: Observer(legacy.clone()),
                seen: seen.clone(),
            })
        )?,
        IRStatus::Changed
    );
    verify_operation(root, ctx)?;
    let output = Operation::get_op_dyn(root, ctx).disp(ctx).to_string();
    let legacy = std::mem::take(&mut *legacy.lock().unwrap());
    let seen = std::mem::take(&mut *seen.lock().unwrap());
    Ok((output, legacy, seen))
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn occurrence_dce_reports_both_same_target_argument_occurrences_before_removal() -> Result<()> {
    let input = r#"
    llvm.func @f: llvm.func <builtin.integer i64 () variadic = false> [] {
      ^entry():
      condition = builtin.constant <builtin.integer <1: i1>> : builtin.integer i1;
      left = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
      right = builtin.constant <builtin.integer <9: i64>> : builtin.integer i64;
      dead = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
      llvm.cond_br if condition ^join(left, dead) else ^join(right, dead)
      ^join(live: builtin.integer i64, unused: builtin.integer i64):
      llvm.return live
    }
    "#;
    let enabled = run_occurrence_pass(input, dce_with_observer, true)?;
    let disabled = run_occurrence_pass(input, dce_with_observer, false)?;
    assert_eq!(enabled.0, disabled.0);
    assert_eq!(
        enabled.1, disabled.1,
        "the legacy stream must remain unchanged"
    );
    assert!(disabled.2.is_empty());
    assert_eq!(
        enabled.2,
        [OccurrenceSeen::Argument {
            index: 1,
            incoming_occurrences: vec![0, 1],
            unequal_live_inputs: true,
        }]
    );
    Ok(())
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn occurrence_cfg_reports_each_chain_merge_before_forwarding_or_erasure() -> Result<()> {
    let input = r#"
    llvm.func @f: llvm.func <builtin.integer i64 () variadic = false> [] {
      ^entry():
      value = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
      llvm.br ^middle(value)
      ^middle(first: builtin.integer i64):
      llvm.br ^last(first)
      ^last(second: builtin.integer i64):
      llvm.return second
    }
    "#;
    let enabled = run_occurrence_pass(input, simplify_cfg_with_observer, true)?;
    let disabled = run_occurrence_pass(input, simplify_cfg_with_observer, false)?;
    assert_eq!(enabled.0, disabled.0);
    assert_eq!(
        enabled.1, disabled.1,
        "the legacy stream must remain unchanged"
    );
    assert!(disabled.2.is_empty());
    assert_eq!(
        enabled.2,
        [
            OccurrenceSeen::Merge {
                actual_count: 1,
                formal_count: 1
            },
            OccurrenceSeen::Merge {
                actual_count: 1,
                formal_count: 1
            },
        ]
    );
    Ok(())
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn occurrence_unreachable_cycle_is_seen_before_successor_uses_are_dropped() -> Result<()> {
    let input = r#"
    llvm.func @f: llvm.func <builtin.integer i64 () variadic = false> [] {
      ^entry():
      value = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
      llvm.return value
      ^dead_first():
      llvm.br ^dead_second()
      ^dead_second():
      llvm.br ^dead_first()
    }
    "#;
    let enabled = run_occurrence_pass(input, simplify_cfg_with_observer, true)?;
    let disabled = run_occurrence_pass(input, simplify_cfg_with_observer, false)?;
    assert_eq!(enabled.0, disabled.0);
    assert_eq!(enabled.1, disabled.1);
    assert!(disabled.2.is_empty());
    assert_eq!(
        enabled.2,
        [
            OccurrenceSeen::DropUses {
                operations: 1,
                successors: 1
            },
            OccurrenceSeen::DropUses {
                operations: 1,
                successors: 1
            },
        ]
    );
    Ok(())
}
