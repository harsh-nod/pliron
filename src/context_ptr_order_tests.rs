fn pointer_order_laws<T: ArenaObj>(pointers: &[Ptr<T>]) {
    use core::cmp::Ordering;
    for &a in pointers {
        assert_eq!(a.cmp(&a), Ordering::Equal);
        for &b in pointers {
            assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
            assert_eq!(a.cmp(&b) == Ordering::Equal, a == b);
            assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
            for &c in pointers {
                if a <= b && b <= c {
                    assert!(a <= c);
                }
                if a < b && b < c {
                    assert!(a < c);
                }
            }
        }
    }
    let forward: alloc::collections::BTreeSet<_> = pointers.iter().copied().collect();
    let backward: alloc::collections::BTreeSet<_> = pointers.iter().rev().copied().collect();
    assert_eq!(forward, backward);
    let mut unique = Vec::new();
    for &pointer in pointers {
        if !unique.contains(&pointer) {
            unique.push(pointer);
        }
    }
    assert_eq!(forward.len(), unique.len());
}

#[test]
fn pointer_order_keeps_equal_slots_from_distinct_contexts_distinct() {
    let mut owner = Context::try_new().unwrap();
    let mut foreign = Context::try_new().unwrap();
    let a = arena_fixture(&mut owner, "owner");
    let b = arena_fixture(&mut foreign, "foreign");
    assert_eq!(a.operation.idx, b.operation.idx);
    assert_eq!(a.region.idx, b.region.idx);
    assert_eq!(a.block.idx, b.block.idx);
    let before = (epoch(&owner), epoch(&foreign));
    pointer_order_laws(&[a.operation, b.operation, a.operation]);
    pointer_order_laws(&[a.region, b.region, a.region]);
    pointer_order_laws(&[a.block, b.block, a.block]);
    assert_ne!(a.operation.cmp(&b.operation), core::cmp::Ordering::Equal);
    assert_ne!(a.region.cmp(&b.region), core::cmp::Ordering::Equal);
    assert_ne!(a.block.cmp(&b.block), core::cmp::Ordering::Equal);
    assert!(is_foreign_context_error(&expect_error(
        a.operation.try_deref(&foreign)
    )));
    assert!(is_foreign_context_error(&expect_error(
        a.region.try_deref(&foreign)
    )));
    assert!(is_foreign_context_error(&expect_error(
        a.block.try_deref(&foreign)
    )));
    assert_eq!((epoch(&owner), epoch(&foreign)), before);
}

#[test]
fn pointer_order_preserves_recycled_arena_generations_without_reviving_stale_keys() {
    let mut context = Context::try_new().unwrap();
    let stale = BasicBlock::new(&mut context, None, vec![]);
    ArenaObj::dealloc(stale, &mut context);
    let replacement = BasicBlock::new(&mut context, None, vec![]);
    let later = BasicBlock::new(&mut context, None, vec![]);
    assert_ne!(stale.idx, replacement.idx);
    let before = epoch(&context);
    pointer_order_laws(&[stale, replacement, later, replacement]);
    assert_ne!(stale.cmp(&replacement), core::cmp::Ordering::Equal);
    assert!(is_dangling_pointer_error(&expect_error(
        stale.try_deref(&context)
    )));
    assert!(replacement.try_deref(&context).is_ok());
    assert!(later.try_deref(&context).is_ok());
    assert_eq!(epoch(&context), before);
}

#[test]
fn pointer_order_is_total_across_slots_generations_and_contexts() {
    let mut contexts = [
        Context::try_new().unwrap(),
        Context::try_new().unwrap(),
        Context::try_new().unwrap(),
    ];
    let mut pointers = Vec::new();
    for context in &mut contexts {
        let first = BasicBlock::new(context, None, vec![]);
        let stale = BasicBlock::new(context, None, vec![]);
        let last = BasicBlock::new(context, None, vec![]);
        ArenaObj::dealloc(stale, context);
        let replacement = BasicBlock::new(context, None, vec![]);
        pointers.extend([replacement, first, stale, last, first]);
    }
    let before = contexts.each_ref().map(epoch);
    pointer_order_laws(&pointers);
    assert_eq!(contexts.each_ref().map(epoch), before);
}

#[test]
fn pointer_order_does_not_borrow_or_mutate_the_pointee() {
    let mut context = Context::try_new().unwrap();
    let first = BasicBlock::new(&mut context, None, vec![]);
    let second = BasicBlock::new(&mut context, None, vec![]);
    let borrowed = first.deref_mut(&context);
    let before = epoch(&context);
    pointer_order_laws(&[second, first, first]);
    assert_eq!(epoch(&context), before);
    drop(borrowed);
    assert!(first.try_deref(&context).is_ok());
}
