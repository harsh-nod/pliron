// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

//! Utilities for attaching / retrieving debug info to / from the IR.

use alloc::{string::ToString, vec::Vec};

use pliron::derive::{attr_interface_impl, pliron_attr};

use crate::{
    attribute::AttributeDict,
    basic_block::BasicBlock,
    builtin::{ATTR_KEY_DEBUG_INFO, attr_interfaces::OutlinedAttr},
    combine::{Parser, attempt, between, parser::char::spaces, sep_by, token},
    context::{Context, Ptr},
    graph::walkers::{self, IRNode, WALKCONFIG_PREORDER_FORWARD},
    identifier::Identifier,
    operation::Operation,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    utils::{table::smalltable::Entry, vec_exns::VecExtns},
};

#[pliron_attr(name = "builtin.debug_info", verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
struct DebugInfoAttr {
    names: Vec<Option<Identifier>>,
}

impl DebugInfoAttr {
    fn get_name(&self, idx: usize) -> Option<Identifier> {
        self.names.get(idx).cloned().flatten()
    }

    fn set_name(&mut self, idx: usize, name: Option<Identifier>) {
        self.names.grow_to(idx + 1, |_| None);
        *self.names.get_mut(idx).unwrap() = name;
    }

    fn insert_name(&mut self, idx: usize, name: Option<Identifier>) {
        self.names.grow_to(idx, |_| None);
        self.names.insert(idx, name);
    }

    fn remove_name(&mut self, idx: usize) {
        if idx < self.names.len() {
            self.names.remove(idx);
        }
    }
}

#[attr_interface_impl]
impl OutlinedAttr for DebugInfoAttr {}

impl Printable for DebugInfoAttr {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "[{}]",
            self.names
                .iter()
                .map(|name| match name {
                    Some(name) => name.disp(ctx).to_string(),
                    None => "?".to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl Parsable for DebugInfoAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        between(
            token('[').skip(spaces()),
            spaces().with(token(']')),
            sep_by(
                attempt(token('?').map(|_| None)).or(Identifier::parser(()).map(Some)),
                token(',').skip(spaces()),
            ),
        )
        .map(|names| DebugInfoAttr { names })
        .parse_stream(state_stream)
        .into()
    }
}

fn set_name_in_attr_map(attributes: &mut AttributeDict, idx: usize, name: Option<Identifier>) {
    match attributes.0.entry(ATTR_KEY_DEBUG_INFO.clone()) {
        Entry::Occupied(mut occupied) => {
            let debug_info = occupied
                .get_mut()
                .downcast_mut::<DebugInfoAttr>()
                .expect("Existing attribute entry for debug info incorrect");
            let name_is_none = name.is_none();
            debug_info.set_name(idx, name);
            // If the debug info attribute has all None entries, remove it from the map.
            if name_is_none && debug_info.names.iter().all(|name| name.is_none()) {
                occupied.remove();
            }
        }
        Entry::Vacant(vacant) => {
            // Only insert a new debug info attribute if there's actually a name to set.
            if name.is_some() {
                let mut debug_info = DebugInfoAttr::default();
                debug_info.set_name(idx, name);
                vacant.insert(debug_info.into());
            }
        }
    }
}

fn insert_name_in_attr_map(attributes: &mut AttributeDict, idx: usize, name: Option<Identifier>) {
    match attributes.0.entry(ATTR_KEY_DEBUG_INFO.clone()) {
        Entry::Occupied(mut occupied) => {
            let debug_info = occupied
                .get_mut()
                .downcast_mut::<DebugInfoAttr>()
                .expect("Existing attribute entry for debug info incorrect");
            let name_is_none = name.is_none();
            debug_info.insert_name(idx, name);
            // If the debug info attribute has all None entries, remove it from the map.
            if name_is_none && debug_info.names.iter().all(|name| name.is_none()) {
                occupied.remove();
            }
        }
        Entry::Vacant(vacant) => {
            // Only insert a new debug info attribute if there's actually a name to set.
            if name.is_some() {
                let mut debug_info = DebugInfoAttr::default();
                debug_info.insert_name(idx, name);
                vacant.insert(debug_info.into());
            }
        }
    }
}

fn remove_name_from_attr_map(attributes: &mut AttributeDict, idx: usize) {
    if let Entry::Occupied(mut occupied) = attributes.0.entry(ATTR_KEY_DEBUG_INFO.clone()) {
        let debug_info = occupied
            .get_mut()
            .downcast_mut::<DebugInfoAttr>()
            .expect("Existing attribute entry for debug info incorrect");
        debug_info.remove_name(idx);
        // If the debug info attribute has no more names, remove it from the map.
        if debug_info.names.iter().all(|name| name.is_none()) {
            occupied.remove();
        }
    }
}

fn get_name_from_attr_map(attributes: &AttributeDict, idx: usize) -> Option<Identifier> {
    attributes
        .get::<DebugInfoAttr>(&ATTR_KEY_DEBUG_INFO)
        .and_then(|debug_info| debug_info.get_name(idx))
}

fn get_name_byte_len_from_attr_map(attributes: &AttributeDict, idx: usize) -> Option<usize> {
    // Scan borrowed keys without initializing the owned lazy key or changing
    // Identifier's map-lookup inference through another Borrow implementation.
    attributes
        .0
        .iter()
        .find(|(key, _)| key.as_ref() == "builtin_debug_info")
        .and_then(|(_, attribute)| attribute.downcast_ref::<DebugInfoAttr>())
        .and_then(|debug_info| debug_info.names.get(idx))
        .and_then(Option::as_ref)
        .map(|name| name.as_ref().len())
}

pub(crate) fn get_operation_result_name_byte_len(
    ctx: &Context,
    op: Ptr<Operation>,
    res_idx: usize,
) -> Option<usize> {
    get_name_byte_len_from_attr_map(&op.deref(ctx).attributes, res_idx)
}

pub(crate) fn get_block_arg_name_byte_len(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    arg_idx: usize,
) -> Option<usize> {
    get_name_byte_len_from_attr_map(&block.deref(ctx).attributes, arg_idx)
}

/// Set the name for a result in an [Operation].
/// Panics if the given `res_idx` is out of range.
pub fn set_operation_result_name(
    ctx: &Context,
    op: Ptr<Operation>,
    res_idx: usize,
    name: Option<Identifier>,
) {
    let op = &mut *op.deref_mut(ctx);
    let num_results = op.get_num_results();
    assert!(res_idx < num_results);

    set_name_in_attr_map(&mut op.attributes, res_idx, name);
}

/// Insert a name for a result in an [Operation] at the given index,
/// shifting existing names at that index and beyond to the right.
/// Panics if the given `res_idx` is out of range (i.e., `res_idx > num_results`).
pub fn insert_operation_result_name(
    ctx: &Context,
    op: Ptr<Operation>,
    res_idx: usize,
    name: Option<Identifier>,
) {
    let op = &mut *op.deref_mut(ctx);
    let num_results = op.get_num_results();
    assert!(res_idx <= num_results);

    insert_name_in_attr_map(&mut op.attributes, res_idx, name);
}

/// Remove the name for a result in an [Operation] at the given index,
/// shifting existing names at beyond that index to the left.
/// Panics if the given `res_idx` is out of range.
pub fn remove_operation_result_name(ctx: &Context, op: Ptr<Operation>, res_idx: usize) {
    let op = &mut *op.deref_mut(ctx);
    let num_results = op.get_num_results();
    assert!(res_idx < num_results);

    remove_name_from_attr_map(&mut op.attributes, res_idx);
}

/// Get name for a result in an [Operation].
pub fn get_operation_result_name(
    ctx: &Context,
    op: Ptr<Operation>,
    res_idx: usize,
) -> Option<Identifier> {
    let op = &*op.deref(ctx);
    get_name_from_attr_map(&op.attributes, res_idx)
}

/// Set the name for an argument in a [BasicBlock].
/// Panics if the given `arg_idx` is out of range.
pub fn set_block_arg_name(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    arg_idx: usize,
    name: Option<Identifier>,
) {
    let block = &mut *block.deref_mut(ctx);
    let num_args = block.get_num_arguments();
    assert!(arg_idx < num_args);

    set_name_in_attr_map(&mut block.attributes, arg_idx, name);
}

/// Insert a name for an argument in a [BasicBlock] at the given index,
/// shifting existing names at that index and beyond to the right.
/// Panics if the given `arg_idx` is out of range (i.e., `arg_idx > num_args`).
pub fn insert_block_arg_name(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    arg_idx: usize,
    name: Option<Identifier>,
) {
    let block = &mut *block.deref_mut(ctx);
    let num_args = block.get_num_arguments();
    assert!(arg_idx <= num_args);

    insert_name_in_attr_map(&mut block.attributes, arg_idx, name);
}

/// Remove the name for an argument in a [BasicBlock] at the given index,
/// shifting existing names at beyond that index to the left.
/// Panics if the given `arg_idx` is out of range.
pub fn remove_block_arg_name(ctx: &Context, block: Ptr<BasicBlock>, arg_idx: usize) {
    let block = &mut *block.deref_mut(ctx);
    let num_args = block.get_num_arguments();
    assert!(arg_idx < num_args);

    remove_name_from_attr_map(&mut block.attributes, arg_idx);
}

/// Get name for an argument in a [BasicBlock].
pub fn get_block_arg_name(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    arg_idx: usize,
) -> Option<Identifier> {
    let block = &*block.deref(ctx);
    get_name_from_attr_map(&block.attributes, arg_idx)
}

/// Recursively erase every [given_name](crate::common_traits::Named::given_name)
/// nested within `op`: Op results, block arguments and block labels.
pub fn erase_given_names(ctx: &mut Context, op: Ptr<Operation>) {
    walkers::uninterruptible::mutable::walk_op(
        ctx,
        &mut (),
        &WALKCONFIG_PREORDER_FORWARD,
        op,
        |ctx: &mut Context, _state: &mut (), node: IRNode| match node {
            IRNode::Operation(op) => {
                let num_results = op.deref(ctx).get_num_results();
                for res_idx in 0..num_results {
                    set_operation_result_name(ctx, op, res_idx, None);
                }
            }
            IRNode::BasicBlock(block) => {
                let num_args = block.deref(ctx).get_num_arguments();
                for arg_idx in 0..num_args {
                    set_block_arg_name(ctx, block, arg_idx, None);
                }
                block.deref_mut(ctx).set_label(None);
            }
            IRNode::Region(_) => {}
        },
    );
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use pliron::derive::pliron_op;

    use crate::{
        basic_block::BasicBlock,
        builtin::{
            op_interfaces::{NOpdsInterface, OneResultInterface},
            types::{IntegerType, Signedness},
        },
        context::Context,
        debug_info::{
            get_block_arg_name, get_operation_result_name, insert_block_arg_name,
            insert_operation_result_name, remove_block_arg_name, remove_operation_result_name,
            set_block_arg_name, set_operation_result_name,
        },
        op::Op,
        operation::{Operation, verify_operation},
        result::Result,
    };

    #[pliron_op(
        name = "test.zero",
        format,
            interfaces = [OneResultInterface, NOpdsInterface<0>],
        verifier = "succ",
    )]
    struct ZeroOp;
    impl ZeroOp {
        pub fn new(ctx: &mut Context) -> Self {
            let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
            ZeroOp {
                op: Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![i64_ty.into()],
                    vec![],
                    vec![],
                    0,
                ),
            }
        }
    }

    fn assert_exact_name_length(ctx: &Context, value: crate::value::Value) {
        use crate::common_traits::Named;
        let expected = value.unique_name(ctx).as_ref().len();
        let epoch = ctx.ir_mutation_attempt_epoch().unwrap();
        for _ in 0..3 {
            assert_eq!(value.unique_name_byte_len(ctx), Some(expected));
            assert_eq!(ctx.ir_mutation_attempt_epoch().unwrap(), epoch);
        }
    }

    #[test]
    fn unique_name_length_tracks_live_result_and_argument_names() {
        let mut ctx = Context::new();
        let ty = IntegerType::get(&ctx, 64, Signedness::Signed);
        let op = Operation::new(
            &mut ctx,
            ZeroOp::get_concrete_op_info(),
            vec![ty.into(), ty.into(), ty.into()],
            vec![],
            vec![],
            0,
        );
        let block = BasicBlock::new(&mut ctx, None, vec![ty.into(), ty.into(), ty.into()]);
        for index in 0..3 {
            let result = op.deref(&ctx).get_result(index);
            let argument = block.deref(&ctx).get_argument(index);
            assert_exact_name_length(&ctx, result);
            assert_exact_name_length(&ctx, argument);
            set_operation_result_name(&ctx, op, index, Some("result_name".try_into().unwrap()));
            set_block_arg_name(
                &ctx,
                block,
                index,
                Some("argument_name".try_into().unwrap()),
            );
            assert_exact_name_length(&ctx, result);
            assert_exact_name_length(&ctx, argument);
            set_operation_result_name(&ctx, op, index, Some("renamed".try_into().unwrap()));
            set_block_arg_name(&ctx, block, index, None);
            assert_exact_name_length(&ctx, result);
            assert_exact_name_length(&ctx, argument);
        }
    }

    #[test]
    fn unique_name_length_handles_long_names_and_shifted_live_values() {
        use crate::builtin::attributes::StringAttr;
        let mut ctx = Context::new();
        let ty = IntegerType::get(&ctx, 64, Signedness::Signed);
        let op = ZeroOp::new(&mut ctx).get_operation();
        let block = BasicBlock::new(&mut ctx, None, vec![ty.into()]);
        let result = op.deref(&ctx).get_result(0);
        let argument = block.deref(&ctx).get_argument(0);
        let long = "x".repeat(8_192);
        set_operation_result_name(&ctx, op, 0, Some(long.as_str().try_into().unwrap()));
        set_block_arg_name(&ctx, block, 0, Some(long.as_str().try_into().unwrap()));
        op.deref_mut(&ctx)
            .attributes
            .set("other".try_into().unwrap(), StringAttr::new("value".into()));
        block
            .deref_mut(&ctx)
            .attributes
            .set("other".try_into().unwrap(), StringAttr::new("value".into()));
        assert!(!op.deref(&ctx).attributes.0.is_inline());
        assert!(!block.deref(&ctx).attributes.0.is_inline());
        assert_exact_name_length(&ctx, result);
        assert_exact_name_length(&ctx, argument);
        Operation::insert_result(op, &ctx, 0, ty.into());
        BasicBlock::insert_argument(block, &ctx, 0, ty.into());
        assert_eq!(result.find_index(&ctx), 1);
        assert_eq!(argument.find_index(&ctx), 1);
        assert_exact_name_length(&ctx, result);
        assert_exact_name_length(&ctx, argument);
        Operation::remove_result(op, &ctx, 0);
        BasicBlock::remove_argument(block, &ctx, 0);
        assert_eq!(result.find_index(&ctx), 0);
        assert_eq!(argument.find_index(&ctx), 0);
        assert_exact_name_length(&ctx, result);
        assert_exact_name_length(&ctx, argument);
    }

    #[test]
    fn borrowed_debug_length_matches_optional_and_malformed_lookup_behavior() {
        use super::{DebugInfoAttr, get_name_byte_len_from_attr_map, get_name_from_attr_map};
        use crate::{attribute::AttributeDict, builtin::attributes::StringAttr};
        let mut attributes = AttributeDict::default();
        assert_eq!(get_name_byte_len_from_attr_map(&attributes, 0), None);
        attributes.set(
            "wrong_key".try_into().unwrap(),
            DebugInfoAttr {
                names: vec![Some("ignored".try_into().unwrap())],
            },
        );
        assert_eq!(get_name_byte_len_from_attr_map(&attributes, 0), None);
        attributes.set(
            "builtin_debug_info".try_into().unwrap(),
            StringAttr::new("wrong type".into()),
        );
        assert_eq!(get_name_byte_len_from_attr_map(&attributes, 0), None);
        assert_eq!(get_name_from_attr_map(&attributes, 0), None);
        attributes = AttributeDict::default();
        attributes.set(
            "builtin_debug_info".try_into().unwrap(),
            DebugInfoAttr {
                names: vec![Some("named".try_into().unwrap()), None],
            },
        );
        for spilled in [false, true] {
            if spilled {
                attributes.set("other".try_into().unwrap(), StringAttr::new("value".into()));
            }
            assert_eq!(attributes.0.is_inline(), !spilled);
            let original = attributes.clone();
            for index in 0..3 {
                let expected =
                    get_name_from_attr_map(&attributes, index).map(|name| name.as_ref().len());
                assert_eq!(
                    get_name_byte_len_from_attr_map(&attributes, index),
                    expected
                );
            }
            assert_eq!(attributes, original);
        }
    }

    #[test]
    fn test_op_result_name() -> Result<()> {
        let mut ctx = Context::new();
        let cop = ZeroOp::new(&mut ctx);
        let op = cop.get_operation();
        set_operation_result_name(&ctx, op, 0, Some("foo".try_into().unwrap()));
        assert_eq!(
            get_operation_result_name(&ctx, op, 0).unwrap(),
            "foo".try_into().unwrap()
        );
        verify_operation(op, &ctx)?;
        Ok(())
    }

    #[test]
    fn test_block_arg_name() -> Result<()> {
        let mut ctx = Context::new();
        let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signed);
        let block = BasicBlock::new(
            &mut ctx,
            Some("entry".try_into().unwrap()),
            vec![i64_ty.into()],
        );
        set_block_arg_name(&ctx, block, 0, Some("foo".try_into().unwrap()));
        assert!(get_block_arg_name(&ctx, block, 0).unwrap() == "foo".try_into().unwrap());
        Ok(())
    }

    #[test]
    fn test_op_result_name_insert_remove_shift() {
        let mut ctx = Context::new();
        let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signed);
        let op = Operation::new(
            &mut ctx,
            ZeroOp::get_concrete_op_info(),
            vec![i64_ty.into(), i64_ty.into(), i64_ty.into()],
            vec![],
            vec![],
            0,
        );

        set_operation_result_name(&ctx, op, 0, Some("r0".try_into().unwrap()));
        set_operation_result_name(&ctx, op, 1, Some("r1".try_into().unwrap()));
        assert_eq!(
            get_operation_result_name(&ctx, op, 0),
            Some("r0".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 1),
            Some("r1".try_into().unwrap())
        );
        assert_eq!(get_operation_result_name(&ctx, op, 2), None);

        // Insert/remove at end should not affect earlier indices.
        insert_operation_result_name(&ctx, op, 2, Some("tail".try_into().unwrap()));
        assert_eq!(
            get_operation_result_name(&ctx, op, 0),
            Some("r0".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 1),
            Some("r1".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 2),
            Some("tail".try_into().unwrap())
        );
        remove_operation_result_name(&ctx, op, 2);
        assert_eq!(
            get_operation_result_name(&ctx, op, 0),
            Some("r0".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 1),
            Some("r1".try_into().unwrap())
        );
        assert_eq!(get_operation_result_name(&ctx, op, 2), None);

        // Insert a placeholder at front, shifting r0 name to index 1.
        insert_operation_result_name(&ctx, op, 0, None);
        assert_eq!(get_operation_result_name(&ctx, op, 0), None);
        assert_eq!(
            get_operation_result_name(&ctx, op, 1),
            Some("r0".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 2),
            Some("r1".try_into().unwrap())
        );

        // Insert a named result at the front.
        insert_operation_result_name(&ctx, op, 0, Some("ins".try_into().unwrap()));
        assert_eq!(
            get_operation_result_name(&ctx, op, 0),
            Some("ins".try_into().unwrap())
        );
        assert_eq!(get_operation_result_name(&ctx, op, 1), None);
        assert_eq!(
            get_operation_result_name(&ctx, op, 2),
            Some("r0".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 3),
            Some("r1".try_into().unwrap())
        );

        // Remove front name; prior index 1 becomes 0 and index 2 becomes 1.
        remove_operation_result_name(&ctx, op, 0);
        assert_eq!(get_operation_result_name(&ctx, op, 0), None);
        assert_eq!(
            get_operation_result_name(&ctx, op, 1),
            Some("r0".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 2),
            Some("r1".try_into().unwrap())
        );

        // Remove the placeholder, moving r0 back to index 0.
        remove_operation_result_name(&ctx, op, 0);
        assert_eq!(
            get_operation_result_name(&ctx, op, 0),
            Some("r0".try_into().unwrap())
        );
        assert_eq!(
            get_operation_result_name(&ctx, op, 1),
            Some("r1".try_into().unwrap())
        );
    }

    #[test]
    fn test_block_arg_name_insert_remove_shift() {
        let mut ctx = Context::new();
        let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signed);
        let block = BasicBlock::new(
            &mut ctx,
            Some("entry".try_into().unwrap()),
            vec![i64_ty.into(), i64_ty.into(), i64_ty.into()],
        );

        set_block_arg_name(&ctx, block, 0, Some("a0".try_into().unwrap()));
        set_block_arg_name(&ctx, block, 1, Some("a1".try_into().unwrap()));
        assert_eq!(
            get_block_arg_name(&ctx, block, 0),
            Some("a0".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 1),
            Some("a1".try_into().unwrap())
        );
        assert_eq!(get_block_arg_name(&ctx, block, 2), None);

        // Insert/remove at end should not affect earlier indices.
        insert_block_arg_name(&ctx, block, 2, Some("tail".try_into().unwrap()));
        assert_eq!(
            get_block_arg_name(&ctx, block, 0),
            Some("a0".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 1),
            Some("a1".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 2),
            Some("tail".try_into().unwrap())
        );
        remove_block_arg_name(&ctx, block, 2);
        assert_eq!(
            get_block_arg_name(&ctx, block, 0),
            Some("a0".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 1),
            Some("a1".try_into().unwrap())
        );
        assert_eq!(get_block_arg_name(&ctx, block, 2), None);

        // Insert a placeholder at front, shifting a0 to index 1.
        insert_block_arg_name(&ctx, block, 0, None);
        assert_eq!(get_block_arg_name(&ctx, block, 0), None);
        assert_eq!(
            get_block_arg_name(&ctx, block, 1),
            Some("a0".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 2),
            Some("a1".try_into().unwrap())
        );

        // Insert a named arg at the front.
        insert_block_arg_name(&ctx, block, 0, Some("ins".try_into().unwrap()));
        assert_eq!(
            get_block_arg_name(&ctx, block, 0),
            Some("ins".try_into().unwrap())
        );
        assert_eq!(get_block_arg_name(&ctx, block, 1), None);
        assert_eq!(
            get_block_arg_name(&ctx, block, 2),
            Some("a0".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 3),
            Some("a1".try_into().unwrap())
        );

        // Remove front name; prior index 1 becomes 0 and index 2 becomes 1.
        remove_block_arg_name(&ctx, block, 0);
        assert_eq!(get_block_arg_name(&ctx, block, 0), None);
        assert_eq!(
            get_block_arg_name(&ctx, block, 1),
            Some("a0".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 2),
            Some("a1".try_into().unwrap())
        );

        // Remove placeholder, moving a0 back to index 0.
        remove_block_arg_name(&ctx, block, 0);
        assert_eq!(
            get_block_arg_name(&ctx, block, 0),
            Some("a0".try_into().unwrap())
        );
        assert_eq!(
            get_block_arg_name(&ctx, block, 1),
            Some("a1".try_into().unwrap())
        );
    }
}
