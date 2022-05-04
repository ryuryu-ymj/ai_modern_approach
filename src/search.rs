use std::{
    collections::{BinaryHeap, HashMap},
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
    rc::Rc,
};

trait Problem {
    type State: Copy + Eq + Hash + Debug;
    type Action: Copy + Eq;
    fn initial(&self) -> Self::State;
    fn actions(&self, s: &Self::State) -> &[Self::Action];
    fn result(&self, s: &Self::State, a: &Self::Action) -> Self::State;
    fn action_cost(&self, s: &Self::State, a: &Self::Action) -> u32;
    fn is_goal(&self, s: &Self::State) -> bool;
}

trait NodeEval<P: Problem, E: NodeEval<P, E>> {
    fn eval(node: &Node<P, E>) -> u32;
}

struct Node<P: Problem, E: NodeEval<P, E>> {
    state: P::State,
    parent: Option<Rc<Node<P, E>>>,
    action: Option<P::Action>,
    path_cost: u32,
    phantom: PhantomData<E>,
}

impl<P, E> Node<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn eval(&self) -> u32 {
        E::eval(self)
    }

    fn root(state: P::State) -> Rc<Node<P, E>> {
        Rc::new(Node {
            state,
            parent: None,
            action: None,
            path_cost: 0,
            phantom: PhantomData,
        })
    }

    fn child(
        state: P::State,
        parent: Rc<Node<P, E>>,
        action: P::Action,
        path_cost: u32,
    ) -> Rc<Node<P, E>> {
        Rc::new(Node {
            state,
            parent: Some(parent),
            action: Some(action),
            path_cost,
            phantom: PhantomData,
        })
    }
}

impl<P, E> PartialEq for Node<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn eq(&self, other: &Self) -> bool {
        self.eval().eq(&other.eval())
    }
}
impl<P, E> Eq for Node<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
}
impl<P, E> PartialOrd for Node<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<P, E> Ord for Node<P, E>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.eval().cmp(&other.eval()).reverse()
    }
}

fn expand<P, E>(problem: &P, node: Rc<Node<P, E>>) -> Vec<Rc<Node<P, E>>>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    let s = &node.state;
    let mut ret = Vec::new();
    for action in problem.actions(&s) {
        let t = problem.result(&s, action);
        let cost = node.path_cost + problem.action_cost(&s, action);
        ret.push(Node::child(t, node.clone(), *action, cost))
    }
    ret
}

fn best_first_search<P, E>(problem: &P) -> Option<Rc<Node<P, E>>>
where
    P: Problem,
    E: NodeEval<P, E>,
{
    let node = Node::root(problem.initial());
    let mut frontier = BinaryHeap::new();
    frontier.push(node);
    let mut reached = HashMap::new();
    while let Some(node) = frontier.pop() {
        if problem.is_goal(&node.state) {
            return Some(node);
        }
        for child in expand(problem, node) {
            let s = child.state;
            if !reached.contains_key(&s) || child.path_cost < reached[&s] {
                reached.insert(s, child.path_cost);
                frontier.push(child);
            }
        }
    }
    None
}

struct UCSEval;
impl<P: Problem> NodeEval<P, UCSEval> for UCSEval {
    fn eval(node: &Node<P, UCSEval>) -> u32 {
        node.path_cost
    }
}

fn uniform_cost_search<P: Problem>(
    problem: &P,
) -> Option<Rc<Node<P, UCSEval>>> {
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

        fn actions(&self, s: &Self::State) -> &[Self::Action] {
            &self.graph[*s]
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
    fn test_shortest_path_problem() {
        let n = 7;
        let es = vec![
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
        let expected_path = vec![0, 2, 3, 5, 4, 6];
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
}
