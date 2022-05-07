pub mod search;

use std::borrow::Cow;

struct EightPuzzle {
    initial: (u8, [u8; 9]),
    goal: (u8, [u8; 9]),
}

impl search::Problem for EightPuzzle {
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

fn main() {
    let p = EightPuzzle {
        initial: (4, [1, 2, 7, 3, 0, 4, 6, 8, 5]),
        goal: (0, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    println!("8 puzzle");
    println!("initial: {:?}", p.initial);
    println!("goal: {:?}", p.goal);

    println!("\nBFS");
    let (node, comp) = search::breadth_first_search(&p);
    match node {
        None => println!("failure"),
        Some(node) => {
            let solution = search::solution(node);
            println!("time: {}, space: {}", comp.time, comp.space);
            for node in solution {
                println!("{:?} {:?}", node.state, node.action);
            }
        }
    }

    println!("\nA* with the number of wrong positions");
    let (node, comp) = search::a_star_search(&p, |node| {
        let mut h = 0;
        for (p, n) in node.state.1.iter().enumerate() {
            if p != *n as usize {
                h += 1;
            }
        }
        h
    });
    match node {
        None => println!("failure"),
        Some(node) => {
            let solution = search::solution(node);
            println!("time: {}, space: {}", comp.time, comp.space);
            for node in solution {
                println!("{:?} {:?}", node.state, node.action);
            }
        }
    }

    println!("\nA* with manhattan distances");
    let (node, comp) = search::a_star_search(&p, |node| {
        let mut h = 0;
        for (p, n) in node.state.1.iter().enumerate() {
            h += (p as u32 % 3).abs_diff(*n as u32 % 3);
            h += (p as u32 / 3).abs_diff(*n as u32 / 3);
        }
        h
    });
    match node {
        None => println!("failure"),
        Some(node) => {
            let solution = search::solution(node);
            println!("time: {}, space: {}", comp.time, comp.space);
            for node in solution {
                println!("{:?} {:?}", node.state, node.action);
            }
        }
    }
}
