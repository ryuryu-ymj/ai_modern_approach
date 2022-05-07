use std::{
    borrow::Cow,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    fmt::Debug,
    hash::Hash,
    rc::Rc,
};

pub struct Complexity {
    pub time: usize,
    pub space: usize,
}

impl Complexity {
    fn new() -> Complexity {
        Complexity { time: 0, space: 0 }
    }
}

pub trait Problem {
    type State: Copy + Eq + Hash + Debug;
    type Action: Copy + Eq + Debug;
    fn initial(&self) -> Self::State;
    fn actions(&self, s: &Self::State) -> Cow<[Self::Action]>;
    fn result(&self, s: &Self::State, a: &Self::Action) -> Self::State;
    fn action_cost(&self, s: &Self::State, a: &Self::Action) -> u32;
    fn is_goal(&self, s: &Self::State) -> bool;
}

pub struct Node<P: Problem> {
    pub state: P::State,
    pub parent: Option<Rc<Node<P>>>,
    pub action: Option<P::Action>,
    pub path_cost: u32,
}

impl<P: Problem> Node<P> {
    fn root(state: P::State) -> Rc<Node<P>> {
        Rc::new(Node {
            state,
            parent: None,
            action: None,
            path_cost: 0,
        })
    }

    fn child(
        state: P::State,
        parent: Rc<Node<P>>,
        action: P::Action,
        path_cost: u32,
    ) -> Rc<Node<P>> {
        Rc::new(Node {
            state,
            parent: Some(parent),
            action: Some(action),
            path_cost,
        })
    }
}

struct OrderdByKey<K, V>
where
    K: Ord,
{
    key: K,
    value: V,
}

impl<K, V> OrderdByKey<K, V>
where
    K: Ord,
{
    fn new(key: K, value: V) -> OrderdByKey<K, V> {
        OrderdByKey { key, value }
    }
}

impl<K, V> PartialEq for OrderdByKey<K, V>
where
    K: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.key.eq(&other.key)
    }
}
impl<K, V> Eq for OrderdByKey<K, V> where K: Ord {}
impl<K, V> PartialOrd for OrderdByKey<K, V>
where
    K: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<K, V> Ord for OrderdByKey<K, V>
where
    K: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key).reverse()
    }
}

fn expand<P>(problem: &P, node: Rc<Node<P>>) -> Vec<Rc<Node<P>>>
where
    P: Problem,
{
    let s = &node.state;
    let mut ret = Vec::new();
    for action in problem.actions(s).iter() {
        let t = problem.result(s, action);
        let cost = node.path_cost + problem.action_cost(s, action);
        ret.push(Node::child(t, node.clone(), *action, cost))
    }
    ret
}

fn best_first_search<P, F>(
    problem: &P,
    eval: F,
) -> (Option<Rc<Node<P>>>, Complexity)
where
    P: Problem,
    F: Fn(&Node<P>) -> u32,
{
    let mut comp = Complexity::new();
    let node = Node::root(problem.initial());
    let mut reached = HashMap::from([(problem.initial(), node.path_cost)]);
    let mut frontier = BinaryHeap::from([OrderdByKey::new(eval(&node), node)]);
    while let Some(node) = frontier.pop() {
        let node = node.value;
        if problem.is_goal(&node.state) {
            return (Some(node), comp);
        }
        for child in expand(problem, node) {
            comp.time += 1;
            let s = child.state;
            if !reached.contains_key(&s) || child.path_cost < reached[&s] {
                reached.insert(s, child.path_cost);
                frontier.push(OrderdByKey::new(eval(&child), child));
            }
        }
        comp.space = comp.space.max(frontier.len());
    }
    (None, comp)
}

pub fn breadth_first_search<P>(problem: &P) -> (Option<Rc<Node<P>>>, Complexity)
where
    P: Problem,
{
    let mut comp = Complexity::new();
    let node = Node::root(problem.initial());
    if problem.is_goal(&node.state) {
        return (Some(node), comp);
    }
    let mut frontier = VecDeque::from([node]);
    let mut reached = HashSet::from([problem.initial()]);
    while let Some(node) = frontier.pop_front() {
        for child in expand(problem, node) {
            comp.time += 1;
            let s = child.state;
            if problem.is_goal(&s) {
                return (Some(child), comp);
            }
            if !reached.contains(&s) {
                reached.insert(s);
                frontier.push_back(child);
            }
        }
        comp.space = comp.space.max(frontier.len());
    }
    (None, comp)
}

pub fn uniform_cost_search<P: Problem>(
    problem: &P,
) -> (Option<Rc<Node<P>>>, Complexity) {
    best_first_search::<P, _>(problem, |node| node.path_cost)
}

pub fn a_star_search<P, F>(
    problem: &P,
    hueristic: F,
) -> (Option<Rc<Node<P>>>, Complexity)
where
    P: Problem,
    F: Fn(&Node<P>) -> u32,
{
    best_first_search::<P, _>(problem, |node| node.path_cost + hueristic(node))
}

pub fn solution<P: Problem>(node: Rc<Node<P>>) -> Vec<Rc<Node<P>>> {
    let mut node = node;
    let mut v = vec![node.clone()];
    while let Some(parent) = &node.parent {
        node = parent.clone();
        v.push(node.clone());
    }
    v.into_iter().rev().collect()
}

#[cfg(test)]
mod test {
    use super::*;

    struct ShortestPathProbelm {
        start: usize,
        goal: usize,
        graph: Vec<Vec<(usize, u32)>>,
    }

    impl Problem for ShortestPathProbelm {
        type State = usize;
        type Action = (usize, u32);

        fn initial(&self) -> Self::State {
            self.start
        }

        fn actions(&self, s: &Self::State) -> Cow<[Self::Action]> {
            Cow::from(&self.graph[*s])
        }

        fn result(&self, _: &Self::State, a: &Self::Action) -> Self::State {
            a.0
        }

        fn action_cost(&self, _: &Self::State, a: &Self::Action) -> u32 {
            a.1
        }

        fn is_goal(&self, s: &Self::State) -> bool {
            s == &self.goal
        }
    }

    #[test]
    fn test_uniform_cost_search() {
        let n = 7;
        let es = [
            (0, 1, 2),
            (0, 2, 5),
            (1, 2, 4),
            (1, 3, 6),
            (1, 4, 10),
            (2, 3, 2),
            (3, 5, 1),
            (4, 5, 3),
            (4, 6, 5),
            (5, 6, 9),
        ];
        let mut g = vec![Vec::new(); n];
        for e in es {
            g[e.0].push((e.1, e.2));
            g[e.1].push((e.0, e.2));
        }
        let problem = ShortestPathProbelm {
            start: 0,
            goal: 6,
            graph: g,
        };
        let expected_path = [0, 2, 3, 5, 4, 6];
        let expected_dst = 16;
        let mut expected = expected_path.into_iter().rev();
        let mut solution = &uniform_cost_search(&problem).0.unwrap();
        assert_eq!(solution.path_cost, expected_dst);
        assert_eq!(solution.state, expected.next().unwrap());
        while let Some(parent) = &solution.parent {
            solution = parent;
            assert_eq!(solution.state, expected.next().unwrap());
        }
    }

    struct EightPuzzle {
        initial: (u8, [u8; 9]),
        goal: (u8, [u8; 9]),
    }

    impl Problem for EightPuzzle {
        type State = (u8, [u8; 9]);
        type Action = u8;

        fn initial(&self) -> Self::State {
            self.initial
        }

        fn is_goal(&self, s: &Self::State) -> bool {
            s == &self.goal
        }

        fn actions(&self, s: &Self::State) -> Cow<[Self::Action]> {
            let mut ret = Vec::new();
            if s.0 > 2 {
                ret.push(0);
            }
            if s.0 % 3 != 2 {
                ret.push(1);
            }
            if s.0 < 6 {
                ret.push(2);
            }
            if s.0 % 3 != 0 {
                ret.push(3);
            }
            Cow::from(ret)
        }

        fn result(&self, s: &Self::State, a: &Self::Action) -> Self::State {
            let p1 = s.0;
            let p2 = match a {
                0 => p1 - 3,     // up
                1 => p1 + 1,     // right
                2 => p1 + 3,     // down
                3 | _ => p1 - 1, // left
            };
            let mut ret = s.clone();
            ret.1.swap(p1.into(), p2.into());
            ret.0 = p2;
            ret
        }

        fn action_cost(&self, _: &Self::State, _: &Self::Action) -> u32 {
            1
        }
    }

    #[test]
    fn test_breadth_first_search() {
        let p = EightPuzzle {
            initial: (4, [1, 2, 7, 3, 0, 4, 6, 8, 5]),
            goal: (0, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        };
        let mut solution = &breadth_first_search(&p).0.unwrap();
        let expected = [1, 0, 3, 2, 1, 2, 3, 0, 0, 3];
        let mut expected = expected.into_iter().rev();
        assert_eq!(solution.action.unwrap(), expected.next().unwrap());
        while let Some(parent) = &solution.parent {
            solution = parent;
            if let Some(action) = solution.action {
                assert_eq!(action, expected.next().unwrap());
            }
        }
    }

    #[test]
    fn test_a_star_search() {
        let p = EightPuzzle {
            initial: (4, [1, 2, 7, 3, 0, 4, 6, 8, 5]),
            goal: (0, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        };
        let mut solution = &a_star_search(&p, |node| {
            let mut h = 0;
            for (i, p) in node.state.1.iter().enumerate() {
                if i != *p as usize {
                    h += 1;
                }
            }
            h
        })
        .0
        .unwrap();
        let expected = [1, 0, 3, 2, 1, 2, 3, 0, 0, 3];
        let mut expected = expected.into_iter().rev();
        assert_eq!(solution.action.unwrap(), expected.next().unwrap());
        while let Some(parent) = &solution.parent {
            solution = parent;
            if let Some(action) = solution.action {
                assert_eq!(action, expected.next().unwrap());
            }
        }
    }
}
