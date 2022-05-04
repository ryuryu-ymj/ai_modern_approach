use std::{
    borrow::Cow,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
    rc::Rc,
};

trait Problem {
    type State: Copy + Eq + Hash + Debug;
    type Action: Copy + Eq;
    fn initial(&self) -> Self::State;
    fn actions(&self, s: &Self::State) -> Cow<[Self::Action]>;
    fn result(&self, s: &Self::State, a: &Self::Action) -> Self::State;
    fn action_cost(&self, s: &Self::State, a: &Self::Action) -> u32;
    fn is_goal(&self, s: &Self::State) -> bool;
}

struct Node<P: Problem> {
    state: P::State,
    parent: Option<Rc<Node<P>>>,
    action: Option<P::Action>,
    path_cost: u32,
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

trait NodeEval<P: Problem, E: NodeEval<P, E>> {
    fn eval(node: &Node<P>) -> u32;
}

struct NodeWrapper<P: Problem, E: NodeEval<P, E>> {
    node: Rc<Node<P>>,
    phantom: PhantomData<E>,
}

impl<P, E> NodeWrapper<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn eval(&self) -> u32 {
        E::eval(&self.node)
    }

    fn from(node: Rc<Node<P>>) -> NodeWrapper<P, E> {
        NodeWrapper {
            node,
            phantom: PhantomData,
        }
    }
}

impl<P, E> PartialEq for NodeWrapper<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn eq(&self, other: &Self) -> bool {
        self.eval().eq(&other.eval())
    }
}
impl<P, E> Eq for NodeWrapper<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
}
impl<P, E> PartialOrd for NodeWrapper<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<P, E> Ord for NodeWrapper<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.eval().cmp(&other.eval()).reverse()
    }
}

fn expand<P>(problem: &P, node: Rc<Node<P>>) -> Vec<Rc<Node<P>>>
where
    P: Problem,
{
    let s = &node.state;
    let mut ret = Vec::new();
    for action in problem.actions(&s).iter() {
        let t = problem.result(&s, action);
        let cost = node.path_cost + problem.action_cost(&s, action);
        ret.push(Node::child(t, node.clone(), *action, cost))
    }
    ret
}

fn best_first_search<P, E>(problem: &P) -> Option<Rc<Node<P>>>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    let node = Node::root(problem.initial());
    let mut frontier: BinaryHeap<NodeWrapper<P, E>> = BinaryHeap::new();
    frontier.push(NodeWrapper::from(node));
    let mut reached = HashMap::new();
    while let Some(node) = frontier.pop() {
        let node = node.node;
        if problem.is_goal(&node.state) {
            return Some(node);
        }
        for child in expand(problem, node) {
            let s = child.state;
            if !reached.contains_key(&s) || child.path_cost < reached[&s] {
                reached.insert(s, child.path_cost);
                frontier.push(NodeWrapper::from(child));
            }
        }
    }
    None
}

fn breadth_first_search<P>(problem: &P) -> Option<Rc<Node<P>>>
where
    P: Problem,
{
    let node = Node::root(problem.initial());
    if problem.is_goal(&node.state) {
        return Some(node);
    }
    let mut frontier = VecDeque::from([node]);
    let mut reached = HashSet::from([problem.initial()]);
    while let Some(node) = frontier.pop_front() {
        for child in expand(problem, node) {
            let s = child.state;
            if problem.is_goal(&s) {
                return Some(child);
            }
            if !reached.contains(&s) {
                reached.insert(s);
                frontier.push_back(child);
            }
        }
    }
    None
}

struct UCSEval;
impl<P: Problem> NodeEval<P, UCSEval> for UCSEval {
    fn eval(node: &Node<P>) -> u32 {
        node.path_cost
    }
}

fn uniform_cost_search<P: Problem>(problem: &P) -> Option<Rc<Node<P>>> {
    return best_first_search::<P, UCSEval>(problem);
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
        let mut solution = &uniform_cost_search(&problem).unwrap();
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
        let mut solution = &breadth_first_search(&p).unwrap();
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
