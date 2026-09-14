// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

mod common;

use std::sync::{Arc, Mutex};

#[cfg(all(not(target_family = "wasm"), panic = "unwind"))]
use pliron::irbuild::listener::Recorder;
#[cfg(target_family = "wasm")]
use wasm_bindgen_test::wasm_bindgen_test;

use common::{ConstantOp, const_ret_in_mod};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    irbuild::{
        inserter::{Inserter, OpInsertionPoint},
        listener::{InsertionListener, RewriteListener},
        observer::{RewriteEvent, RewriteObserver},
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::LinkedList,
    op::{Op, verify_op},
    operation::Operation,
    region::Region,
    r#type::{TypeHandle, Typed},
    value::Value,
};

type Trace = Arc<Mutex<Vec<(&'static str, &'static str)>>>;

#[derive(Default)]
struct TimingListener {
    trace: Trace,
}

impl TimingListener {
    fn record(&self, kind: &'static str) {
        self.trace.lock().unwrap().push(("listener", kind));
    }
}

impl InsertionListener for TimingListener {
    fn notify_operation_inserted(&mut self, ctx: &Context, op: Ptr<Operation>) {
        assert!(op.deref(ctx).get_parent_block().is_some());
        self.record("insert");
    }
    fn notify_block_inserted(&mut self, _ctx: &Context, _block: Ptr<BasicBlock>) {
        panic!("unexpected block insertion in operation-only fixture");
    }
}

impl RewriteListener for TimingListener {
    fn notify_operation_erasure(&mut self, ctx: &Context, op: Ptr<Operation>) {
        assert!(op.deref(ctx).get_parent_block().is_some());
        self.record("erase");
    }
    fn notify_operation_unlinking(&mut self, ctx: &Context, op: Ptr<Operation>) {
        assert!(op.deref(ctx).get_parent_block().is_some());
        self.record("unlink");
    }
    fn notify_value_use_replacement(&mut self, ctx: &Context, old: Value, new: Value) {
        assert!(old.is_used(ctx));
        assert!(!new.is_used(ctx));
        self.record("value");
    }
    fn notify_value_type_change(
        &mut self,
        ctx: &Context,
        value: Value,
        old_type: TypeHandle,
        new_type: TypeHandle,
    ) {
        assert_eq!(value.get_type(ctx), old_type);
        assert_ne!(old_type, new_type);
        self.record("type");
    }
    fn notify_block_erasure(&mut self, _ctx: &Context, _block: Ptr<BasicBlock>) {
        panic!("unexpected block erasure in operation-only fixture");
    }
    fn notify_region_erasure(&mut self, _ctx: &Context, _region: Ptr<Region>) {
        panic!("unexpected region erasure in operation-only fixture");
    }
    fn notify_block_unlinking(&mut self, _ctx: &Context, _block: Ptr<BasicBlock>) {
        panic!("unexpected block unlink in operation-only fixture");
    }
}

struct TimingObserver {
    trace: Trace,
}

impl RewriteObserver for TimingObserver {
    fn observe(&mut self, ctx: &Context, event: RewriteEvent) {
        let kind = match event {
            RewriteEvent::OperationInserted(op) => {
                assert!(op.deref(ctx).get_parent_block().is_some());
                "insert"
            }
            RewriteEvent::OperationErased(op) => {
                assert!(op.deref(ctx).get_parent_block().is_some());
                "erase"
            }
            RewriteEvent::OperationUnlinked(op) => {
                assert!(op.deref(ctx).get_parent_block().is_some());
                "unlink"
            }
            RewriteEvent::ValueReplaced { old, new } => {
                assert!(old.is_used(ctx));
                assert!(!new.is_used(ctx));
                "value"
            }
            RewriteEvent::ValueTypeChanged {
                value,
                old_type,
                new_type,
            } => {
                assert_eq!(value.get_type(ctx), old_type);
                assert_ne!(old_type, new_type);
                "type"
            }
            RewriteEvent::OperationReplaced { old, new } => {
                assert!(old.deref(ctx).get_result(0).is_used(ctx));
                assert!(!new.deref(ctx).get_result(0).is_used(ctx));
                "operation"
            }
            _ => panic!("unexpected subtree event in operation-only fixture"),
        };
        self.trace.lock().unwrap().push(("observer", kind));
    }
}

fn timed_rewriter(trace: &Trace) -> IRRewriter<TimingListener> {
    let mut rewriter = IRRewriter::default();
    rewriter.set_listener(TimingListener {
        trace: Arc::clone(trace),
    });
    rewriter.set_observer(Some(Box::new(TimingObserver {
        trace: Arc::clone(trace),
    })));
    rewriter
}

fn take_trace(trace: &Trace) -> Vec<(&'static str, &'static str)> {
    std::mem::take(&mut *trace.lock().unwrap())
}

#[cfg_attr(not(target_family = "wasm"), test)]
#[cfg_attr(target_family = "wasm", wasm_bindgen_test)]
fn mutation_callbacks_have_documented_live_state_and_listener_order() -> pliron::result::Result<()>
{
    let ctx = &mut Context::new();
    let (module, _, original, ret) = const_ret_in_mod(ctx)?;
    let trace = Trace::default();
    let mut rewriter = timed_rewriter(&trace);
    let original_op = original.get_operation();
    let replacement = ConstantOp::new(ctx, 8);
    let replacement_op = replacement.get_operation();
    rewriter.set_insertion_point_before_operation(ret.get_operation());
    rewriter.insert_operation(ctx, replacement_op);
    assert_eq!(
        take_trace(&trace),
        [("listener", "insert"), ("observer", "insert")]
    );

    rewriter.move_operation(
        ctx,
        original_op,
        OpInsertionPoint::BeforeOperation(ret.get_operation()),
    );
    assert_eq!(original_op.deref(ctx).get_prev(), Some(replacement_op));
    assert_eq!(
        take_trace(&trace),
        [
            ("observer", "unlink"),
            ("listener", "unlink"),
            ("listener", "insert"),
            ("observer", "insert"),
        ]
    );

    let value = original.get_result(ctx);
    let old_type = value.get_type(ctx);
    let new_type = IntegerType::get(ctx, 32, Signedness::Signed).into();
    rewriter.set_value_type(ctx, value, new_type);
    assert_eq!(value.get_type(ctx), new_type);
    assert_eq!(
        take_trace(&trace),
        [("observer", "type"), ("listener", "type")]
    );
    rewriter.set_value_type(ctx, value, new_type);
    assert!(take_trace(&trace).is_empty());
    rewriter.set_value_type(ctx, value, old_type);
    assert_eq!(
        take_trace(&trace),
        [("observer", "type"), ("listener", "type")]
    );

    rewriter.replace_value_uses_with(ctx, value, replacement.get_result(ctx));
    assert_eq!(
        ret.get_operation().deref(ctx).get_operand(0),
        replacement.get_result(ctx)
    );
    assert!(!value.is_used(ctx));
    assert_eq!(
        take_trace(&trace),
        [("observer", "value"), ("listener", "value")]
    );
    rewriter.replace_value_uses_with(ctx, value, value);
    assert!(take_trace(&trace).is_empty());

    let final_op = ConstantOp::new(ctx, 9);
    rewriter.insert_operation(ctx, final_op.get_operation());
    assert_eq!(
        take_trace(&trace),
        [("listener", "insert"), ("observer", "insert")]
    );
    rewriter.replace_operation(ctx, replacement_op, final_op.get_operation());
    assert_eq!(
        take_trace(&trace),
        [
            ("observer", "operation"),
            ("observer", "value"),
            ("listener", "value"),
            ("observer", "erase"),
            ("listener", "erase"),
        ]
    );
    assert!(replacement_op.try_deref(ctx).is_err());
    assert_eq!(
        ret.get_operation().deref(ctx).get_operand(0),
        final_op.get_result(ctx)
    );

    rewriter.erase_operation(ctx, original_op);
    assert_eq!(
        take_trace(&trace),
        [("observer", "erase"), ("listener", "erase")]
    );
    assert!(original_op.try_deref(ctx).is_err());
    verify_op(&module, ctx)?;
    Ok(())
}

#[cfg_attr(not(target_family = "wasm"), test)]
#[cfg_attr(target_family = "wasm", wasm_bindgen_test)]
fn prebuilt_subtree_insertion_reports_only_its_root_and_direct_mutation_is_unobserved()
-> pliron::result::Result<()> {
    let ctx = &mut Context::new();
    let (original, function, _, _) = const_ret_in_mod(ctx)?;
    let destination = ModuleOp::new(ctx, "destination".try_into().unwrap());
    let trace = Trace::default();
    let mut rewriter = timed_rewriter(&trace);
    function.get_operation().unlink(ctx);
    assert!(take_trace(&trace).is_empty());
    rewriter.set_insertion_point_to_block_end(destination.get_body(ctx, 0));
    rewriter.insert_operation(ctx, function.get_operation());
    // No nested block or operation insertion callbacks are synthesized.
    assert_eq!(
        take_trace(&trace),
        [("listener", "insert"), ("observer", "insert")]
    );
    verify_op(&original, ctx)?;
    verify_op(&destination, ctx)?;
    Ok(())
}

#[cfg(all(not(target_family = "wasm"), panic = "unwind"))]
struct PanicOnInsert;

#[cfg(all(not(target_family = "wasm"), panic = "unwind"))]
impl RewriteObserver for PanicOnInsert {
    fn observe(&mut self, _ctx: &Context, event: RewriteEvent) {
        if matches!(event, RewriteEvent::OperationInserted(_)) {
            panic!("injected observer panic");
        }
    }
}

#[test]
#[cfg(all(not(target_family = "wasm"), panic = "unwind"))]
fn insertion_observer_panic_does_not_roll_back_ir_or_legacy_listener() -> pliron::result::Result<()>
{
    let ctx = &mut Context::new();
    let (module, _, _, ret) = const_ret_in_mod(ctx)?;
    let inserted = ConstantOp::new(ctx, 8).get_operation();
    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.set_observer(Some(Box::new(PanicOnInsert)));
    rewriter.set_insertion_point_before_operation(ret.get_operation());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rewriter.insert_operation(ctx, inserted);
    }));
    assert!(result.is_err());
    assert!(inserted.deref(ctx).get_parent_block().is_some());
    assert!(matches!(rewriter.get_listener().events.as_slice(),
        [pliron::irbuild::listener::RecorderEvent::InsertedOperation(op)] if *op == inserted));
    // Reconciliation is the caller's responsibility after catching the panic.
    verify_op(&module, ctx)?;
    Ok(())
}
