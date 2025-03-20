pub mod comp;
mod compile;
pub mod det;
mod dot;
pub mod inter;

use std::error::Error;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
};

use crate::{
    alphabet::{AlphabetPartitionMap, CharRange},
    SmtChar, SmtString,
};

extern crate dot as dotlib;

pub use compile::compile;

/// The type of a transition in a nondeterministic finite automaton.
/// Every transition in an automaton is labeled with a type that determines the behavior of the transition.
/// The type can be a character range, a negated character range, or an epsilon transition.
/// For a character range transition, the transition is taken if the input character is in the range.
/// For a negated character range transition, the transition is taken if the input character is not in the range.
/// An epsilon transition is taken without consuming any input.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub enum TransitionType {
    /// A transition that is taken if the input character is in the given range.
    Range(CharRange),
    /// A transition that is taken if the input character is not in the given range.
    NotRange(CharRange),
    /// An epsilon transition that is taken without consuming any input.
    Epsilon,
}

impl TransitionType {
    /// Returns true if the transition is an epsilon transition.
    pub fn is_epsilon(&self) -> bool {
        matches!(self, TransitionType::Epsilon)
    }

    /// Returns true if the transition is an empty transition, i.e. a transition that can never be taken.
    pub fn is_empty(&self) -> bool {
        matches!(self, TransitionType::Epsilon)
    }
}

/// A transition from one state to another.
/// The transition can be of different types, e.g. a character range or an epsilon transition.
/// The destination state is stored as an index that is unique within the automaton.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Transition {
    /// The type of the transition.
    label: TransitionType,
    /// The index of the destination state.
    destination: usize,
}

impl Transition {
    pub fn new(label: TransitionType, destination: usize) -> Self {
        Self { label, destination }
    }

    /// Returns the destination state of the transition.
    pub fn get_dest(&self) -> usize {
        self.destination
    }

    /// Returns the type of the transition.
    pub fn get_type(&self) -> &TransitionType {
        &self.label
    }

    /// Returns if the transition is an epsilon transition.
    pub fn is_epsilon(&self) -> bool {
        matches!(self.label, TransitionType::Epsilon)
    }
}

/// A state in a nondeterministic finite automaton.
/// A state is merely a collection of transitions to other states.
/// In a nondeterministic automaton, a state can have multiple transitions with the same input leading to different states.
/// If the state has no transitions, it is a dead end state.
/// If the state is a final state, it accepts the input.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct State {
    transitions: Vec<Transition>,
}

impl State {
    /// Returns an iterator over the transitions of the state.
    fn transitions(&self) -> impl Iterator<Item = &Transition> {
        self.transitions.iter()
    }

    /// Adds a transition to the state.
    fn add_transition(&mut self, label: TransitionType, destination: StateId) {
        let transition = Transition { label, destination };
        self.transitions.push(transition);
    }

    /// Consumes the input character and returns the set of states that can be reached from this state.
    fn consume(&self, input: SmtChar) -> HashSet<StateId> {
        let mut res = HashSet::new();
        for t in self.transitions() {
            match t.label {
                TransitionType::Range(r) if r.contains(input) => {
                    res.insert(t.destination);
                }
                TransitionType::NotRange(r) if !r.contains(input) => {
                    res.insert(t.destination);
                }
                _ => {}
            }
        }
        res
    }

    /// Checks if the state is deterministic.
    /// A state is deterministic if it has at most one transition for each input character.
    /// A state is not deterministic if it has multiple transitions for the same input character or an epsilon transition.
    pub fn is_det(&self) -> bool {
        let mut map = AlphabetPartitionMap::empty();
        for t in self.transitions.iter() {
            match t.label {
                TransitionType::Range(r) => {
                    if map.insert(r, t.get_dest()).is_err() {
                        return false;
                    }
                }
                TransitionType::NotRange(r) => {
                    for r in r.complement() {
                        if map.insert(r, t.get_dest()).is_err() {
                            return false;
                        }
                    }
                }
                _ => return false,
            }
        }
        true
    }
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct StateNotFound(pub StateId);

impl Display for StateNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State not found: {}", self.0)
    }
}

impl Error for StateNotFound {}

/// Every state in a nondeterministic automaton is identified by a unique index.
pub type StateId = usize;

/// A nondeterministic finite automaton.
/// The automaton consists of a collection of states, an initial state, and a set of final states.
#[derive(Debug, Clone, Default)]
pub struct NFA {
    states: Vec<State>,
    initial: Option<StateId>,
    finals: HashSet<StateId>,

    /// Flag that indicates whether the automaton is trim.
    /// This is true if the automaton has been trimmed, i.e. all states are reachable from the initial state and all states can reach a final state.
    /// This set to false whenever a new state is added to the automaton.
    /// Note that adding a transition does not affect the trim flag.
    /// The empty automaton  is considered trim.
    trim: bool,
    /// Flag that indicates whether the automaton is epsilon-free.
    /// This is true if the automaton has no epsilon transitions.
    efree: bool,
}

impl NFA {
    /// Create a new, empty nondeterministic finite automaton.
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            initial: None,
            finals: HashSet::new(),
            trim: true,
            efree: true,
        }
    }

    /// Add a new state to the automaton and return its index.
    pub fn new_state(&mut self) -> StateId {
        let id = self.states.len();
        self.states.push(State {
            transitions: Vec::new(),
        });
        self.trim = false;
        id
    }

    /// Set the initial state of the automaton.
    /// The index must be a valid state index, otherwise an error is returned.
    pub fn set_initial(&mut self, state: StateId) -> Result<(), StateNotFound> {
        if state < self.states.len() {
            self.initial = Some(state);
            Ok(())
        } else {
            Err(StateNotFound(state))
        }
    }

    /// Returns the initial state of the automaton, if it exists.
    pub fn initial(&self) -> Option<StateId> {
        self.initial
    }

    /// Add a state to the set of final states.
    /// The index must be a valid state index, otherwise an error is returned.
    pub fn add_final(&mut self, state: StateId) -> Result<(), StateNotFound> {
        if state < self.states.len() {
            self.finals.insert(state);
            Ok(())
        } else {
            Err(StateNotFound(state))
        }
    }

    /// Returns an iterator over the final states of the automaton.
    pub fn finals(&self) -> impl Iterator<Item = StateId> + '_ {
        self.finals.iter().copied()
    }

    /// Returns if a state is a final state.
    /// Invalid indices are not considered final states.
    pub fn is_final(&self, state: StateId) -> bool {
        self.finals.contains(&state)
    }

    /// Add a transition from one state to another.
    /// The indices must be valid state indices, otherwise an error is returned.
    pub fn add_transition(
        &mut self,
        from: StateId,
        to: StateId,
        label: TransitionType,
    ) -> Result<(), StateNotFound> {
        if to >= self.states.len() {
            return Err(StateNotFound(to));
        }
        self.efree &= !label.is_epsilon();
        if let Some(s) = self.states.get_mut(from) {
            s.add_transition(label, to);
            Ok(())
        } else {
            Err(StateNotFound(from))
        }
    }

    /// Returns the number of states in the automaton.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Returns the number of transitions in the automaton.
    /// This is the sum of the number of transitions of each state.
    /// Note that this require a linear scan over all states and transitions.
    pub fn num_transitions(&self) -> usize {
        self.states.iter().map(|s| s.transitions.len()).sum()
    }

    /// Returns an iterator over the states of the automaton.
    pub fn states(&self) -> impl Iterator<Item = StateId> {
        0..self.states.len()
    }

    /// Returns an iterator over the transitions of the automaton.
    /// The iterator yields pairs of a state and its outgoing transitions.
    pub fn transitions(&self) -> impl Iterator<Item = (StateId, &Vec<Transition>)> {
        self.states
            .iter()
            .enumerate()
            .map(|(i, s)| (i, &s.transitions))
    }

    /// Returns an iterator over the transitions from a state.
    /// If the state is not a valid state index, an error is returned.
    pub fn transitions_from(
        &self,
        state: StateId,
    ) -> Result<impl Iterator<Item = &Transition>, StateNotFound> {
        if state < self.states.len() {
            Ok(self.states[state].transitions())
        } else {
            Err(StateNotFound(state))
        }
    }

    /// Returns a map from states to the set of their predecessors.
    /// A state p has predecessor q if there is a transition from q to p.
    pub fn predecessors(&self) -> HashMap<StateId, HashSet<StateId>> {
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        let mut preds = HashMap::new();
        if let Some(q0) = self.initial {
            queue.push_back(q0);
            seen.insert(q0);
        }
        while let Some(q) = queue.pop_front() {
            for t in self.transitions_from(q).unwrap() {
                let dest = t.destination;
                if !seen.contains(&dest) {
                    queue.push_back(dest);
                    seen.insert(dest);
                }
                preds.entry(dest).or_insert(HashSet::new()).insert(q);
            }
        }
        preds
    }

    /// Checks if the automaton is deterministic.
    /// An automaton is deterministic if all states are deterministic.
    /// A state is deterministic if it has at most one transition for each input character and no epsilon transitions.
    /// Checking for determinism is done in O(|V| + |E|) time, where |V| is the number of states and |E| is the number of transitions.
    pub fn is_det(&self) -> bool {
        self.states.iter().all(|s| s.is_det())
    }

    /// Trims the automaton by removing unreachable and dead states.
    /// This ensures that all states are reachable from the initial state AND can reach a final state.
    /// Runs in **O(V + E)** time using two BFS traversals.
    pub fn trim(&self) -> Self {
        if self.trim {
            return self.clone();
        }
        if self.finals.is_empty() || self.initial.is_none() {
            return Self::new(); // Empty automaton if no final or initial state
        }

        let initial = self.initial.unwrap();

        // Forward BFS to find reachable states
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(initial);
        reachable.insert(initial);

        while let Some(state) = queue.pop_front() {
            for t in self.states[state].transitions() {
                let dest = t.destination;
                if reachable.insert(dest) {
                    // Insert returns true if newly inserted
                    queue.push_back(dest);
                }
            }
        }

        // Reverse BFS from final states to find states that can reach a final state
        // We are left with `useful` states that are reachable from the initial state and can reach a final state
        let mut useful = HashSet::new();
        let mut queue = VecDeque::new();
        let predecessors = self.predecessors();

        for &f in &self.finals {
            if reachable.contains(&f) {
                // Only consider reachable final states
                queue.push_back(f);
                useful.insert(f);
            }
        }

        while let Some(state) = queue.pop_front() {
            if let Some(preds) = predecessors.get(&state) {
                for &src in preds {
                    if reachable.contains(&src) && useful.insert(src) {
                        queue.push_back(src);
                    }
                }
            }
        }

        //  Build the new trimmed automaton
        let mut aut = Self::new();
        let mut state_map = HashMap::new();

        for &state in useful.iter() {
            let new_state = aut.new_state();
            state_map.insert(state, new_state);
            if self.finals.contains(&state) {
                aut.add_final(new_state).unwrap();
            }
            if state == initial {
                aut.set_initial(new_state).unwrap();
            }
        }

        // Copy  transitions
        for &state in useful.iter() {
            let new_state = state_map[&state];
            for t in self.states[state].transitions() {
                let dest = t.destination;
                if useful.contains(&dest) {
                    let new_dest = state_map[&dest];
                    aut.add_transition(new_state, new_dest, t.label).unwrap();
                }
            }
        }

        aut
    }

    /// Returns the epsilon closure of a state.
    /// The epsilon closure of a state is the set of states that can be reached from the state by following epsilon transitions.
    /// If the state is not a valid state index, an error is returned.
    pub fn epsilon_closure(&self, state: StateId) -> Result<HashSet<StateId>, StateNotFound> {
        if state >= self.states.len() {
            return Err(StateNotFound(state));
        }
        let mut closure = HashSet::new();
        let mut stack = Vec::new();
        stack.push(state);
        while let Some(s) = stack.pop() {
            closure.insert(s);
            for t in self.states[s].transitions.iter() {
                if t.is_epsilon() {
                    let dest = t.destination;
                    if !closure.contains(&dest) {
                        stack.push(dest);
                    }
                }
            }
        }
        Ok(closure)
    }

    /// Performs the standard epsilon removal algorithm on the automaton.
    /// The result is a new automaton that accepts the same language as this automaton but has no epsilon transitions.
    /// If the automaton is already epsilon-free, this is a no-op.
    /// The resulting automaton is also trim as a side effect.
    ///
    /// The algorithm works as follows:
    /// We do a breadth-first search starting from the initial state and compute the epsilon closure of each state.
    /// The epsilon closure of a state is the set of states that can be reached from the state by following epsilon transitions.
    /// We then add transitions for all non-epsilon transitions from the state to the destination states of the epsilon closure.
    /// If any state in the epsilon closure is a final state, we mark the new state as final.
    ///
    /// The algorithm runs in O(|V| + |E|) time, where |V| is the number of states and |E| is the number of transitions.
    pub fn eliminate_epsilon(self) -> Self {
        if self.efree {
            return self;
        }
        // We do a breadth-first search starting from the initial state
        // and compute the epsilon closure of each state.
        // The epsilon closure of a state is the set of states that can be reached from the state by following epsilon transitions.
        // We then add transitions for all non-epsilon transitions from the state to the destination states of the epsilon closure.
        // If any state in the epsilon closure is a final state, we mark the new state as final.

        let mut queue = VecDeque::new();
        let mut aut = Self::new();
        // The automaton is epsilon-free after eliminating epsilon transitions. As a side effect, the automaton is also trimmed.
        aut.efree = true;
        aut.trim = true;
        let mut old_to_new = HashMap::new();

        if let Some(q0) = self.initial {
            let q0_new = aut.new_state();
            aut.set_initial(q0_new).unwrap();
            queue.push_back(q0);
            old_to_new.insert(q0, q0_new);
        } else {
            return aut;
        }

        while let Some(q) = queue.pop_front() {
            // we ensure that a state is in the old_to_new map before adding it to the queue
            let q_new = *old_to_new.get(&q).unwrap();
            // obtain the epsilon closure of the current state
            let closure = self.epsilon_closure(q).unwrap();
            for q_i in closure {
                if self.finals.contains(&q_i) {
                    aut.add_final(q_new).unwrap();
                }
                // For all non-epsilon transitions from the state, add a transition to the destination state
                for t in self.states[q_i].transitions() {
                    if !t.is_epsilon() {
                        let q_j = t.destination;
                        let q_j_new = match old_to_new.get(&q_j) {
                            Some(q_j_new) => *q_j_new,
                            None => {
                                let q_j_new = aut.new_state();
                                old_to_new.insert(q_j, q_j_new);
                                queue.push_back(q_j);
                                q_j_new
                            }
                        };
                        aut.add_transition(q_new, q_j_new, t.label).unwrap();
                    }
                }
            }
        }
        aut
    }

    /// Returns the longest non-cyclic path in the automaton from the initial state to a final state, if it exists.
    /// The length of the path is the number of transitions in the path and, therefore, the number of characters in the longest word accepted by the automaton.
    /// Epsilon transitions are not counted in the path length.
    /// If the automaton is empty or contains a cycle on an accepting path, this returns None.
    pub fn longest_path(&self) -> Option<usize> {
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        queue.push_back((self.initial?, 0));
        seen.insert(self.initial?);

        let mut longest = -1;
        while let Some((q, len)) = queue.pop_front() {
            for t in self.transitions_from(q).unwrap() {
                let d = if t.is_epsilon() { 0 } else { 1 };
                let dest = t.get_dest();
                if !seen.contains(&dest) {
                    seen.insert(dest);
                    queue.push_back((dest, len + d));
                    if self.is_final(dest) {
                        longest = longest.max(len + d);
                    }
                } else {
                    // If we have a cycle, we can't have a longest path
                    return None;
                }
            }
        }
        if longest >= 0 {
            Some(longest as usize)
        } else {
            None
        }
    }

    /// Returns the shortest path in the automaton from the initial state to a final state, if it exists.
    /// The length of the path is the number of transitions in the path and, therefore, the number of characters in the shortest word accepted by the automaton.
    /// Epsilon transitions are not counted in the path length.
    /// If the automaton is empty, this returns None.
    pub fn shortest_path(&self) -> Option<usize> {
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        queue.push_back((self.initial?, 0));
        seen.insert(self.initial?);
        while let Some((q, len)) = queue.pop_front() {
            if self.is_final(q) {
                return Some(len);
            }
            for t in self.transitions_from(q).unwrap() {
                let d = if t.is_epsilon() { 0 } else { 1 };
                let dest = t.get_dest();
                if !seen.contains(&dest) {
                    seen.insert(dest);
                    if t.is_epsilon() {
                        queue.push_front((dest, len));
                    } else {
                        queue.push_back((dest, len + d));
                    }
                }
            }
        }
        None
    }

    /// Returns whether the automaton is empty.
    /// An automaton is empty if either
    ///
    /// - it has no final states,
    /// - it has no initial state, or
    /// - there is no path from the initial state to a final state.
    pub fn is_empty(&self) -> bool {
        if self.trim {
            return self.finals.is_empty() || self.initial.is_none();
        } else {
            // Check if there is a path from the initial state to a final state
            // If a path exists, then also a shortest path exists
            self.shortest_path().is_none()
        }
    }

    /// Returns the set of states that can be reached from the initial state by consuming the given word.
    pub fn run(&self, word: &SmtString) -> HashSet<StateId> {
        let mut current = HashSet::new();

        if let Some(initial) = self.initial {
            current = self.epsilon_closure(initial).unwrap();
        }

        for c in word.iter() {
            let mut next_states = HashSet::new();
            for s in current {
                for reached in self.states[s].consume(*c) {
                    next_states.extend(self.epsilon_closure(reached).unwrap());
                }
            }
            current = next_states;
        }

        current
    }

    /// Returns if the automaton accepts the given word.
    /// A word is accepted if there is a path from the initial state to a final state by consuming the word.
    pub fn accepts(&self, word: &SmtString) -> bool {
        let reached = self.run(word);
        !reached.is_disjoint(&self.finals)
    }

    /// Returns the DOT representation of the automaton.
    /// The DOT representation can be used to visualize the automaton using Graphviz.
    pub fn dot(&self) -> String {
        let mut buf = Vec::new();
        dotlib::render(self, &mut buf).unwrap();
        String::from_utf8(buf).expect("Failed to convert DOT to string")
    }
}

impl Display for TransitionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransitionType::Range(r) => write!(f, "{}", r),
            TransitionType::NotRange(r) => write!(f, "not({})", r),
            TransitionType::Epsilon => write!(f, ""),
        }
    }
}

impl Display for NFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NFA {{")?;
        writeln!(f, "\tStates:")?;
        for (i, state) in self.states.iter().enumerate() {
            write!(f, "\t\t{}: ", i)?;
            for t in state.transitions() {
                write!(f, "{} -> {}, ", t.label, t.destination)?;
            }
            writeln!(f)?;
        }
        if let Some(q0) = self.initial {
            writeln!(f, "\tInitial: {q0}")?;
        } else {
            writeln!(f, "\tInitial: None")?;
        }
        writeln!(f, "\tFinals: {:?}", self.finals)?;
        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_invalid_initial_state() {
        let mut a = NFA::new();
        let result = a.set_initial(0);
        assert_eq!(result, Err(StateNotFound(0)));
    }

    #[test]
    fn test_valid_initial_state() {
        let mut a = NFA::new();
        let state = a.new_state();
        let result = a.set_initial(state);
        assert!(result.is_ok());
        assert_eq!(a.initial, Some(state));
    }

    #[test]
    fn test_invalid_final_state() {
        let mut a = NFA::new();
        let result = a.add_final(0);
        assert_eq!(result, Err(StateNotFound(0)));
    }

    #[test]
    fn test_valid_final_state() {
        let mut a = NFA::new();
        let state = a.new_state();
        let result = a.add_final(state);
        assert!(result.is_ok());
        assert!(a.finals.contains(&state));
    }

    #[test]
    fn test_invalid_transition_from() {
        let mut a = NFA::new();
        let s = a.new_state();
        let unknown_state = a.num_states() + 1;
        let result = a.add_transition(unknown_state, s, TransitionType::Epsilon);
        assert_eq!(result, Err(StateNotFound(unknown_state)));
    }

    #[test]
    fn test_invalid_transition_to() {
        let mut a = NFA::new();
        let state = a.new_state();
        let result = a.add_transition(state, 1, TransitionType::Epsilon);
        assert_eq!(result, Err(StateNotFound(1)));
    }

    #[test]
    fn test_valid_transition() {
        let mut a = NFA::new();
        let state1 = a.new_state();
        let state2 = a.new_state();
        let result = a.add_transition(state1, state2, TransitionType::Epsilon);
        assert!(result.is_ok());
        assert_eq!(a.states[state1].transitions.len(), 1);
        assert_eq!(a.states[state1].transitions[0].destination, state2);
        assert_eq!(
            a.states[state1].transitions[0].label,
            TransitionType::Epsilon
        );
    }

    /* Tests for epsilon removal */

    #[test]
    fn test_epsilon_closure() {
        let mut a = NFA::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();
        a.add_transition(s0, s1, TransitionType::Epsilon).unwrap();
        a.add_transition(s1, s2, TransitionType::Epsilon).unwrap();
        a.add_transition(s2, s3, TransitionType::Epsilon).unwrap();
        let result = a.epsilon_closure(s0).unwrap();
        assert_eq!(result.len(), 4);
        assert!(result.contains(&s0));
        assert!(result.contains(&s1));
        assert!(result.contains(&s2));
        assert!(result.contains(&s3));
    }

    #[test]
    fn test_elim_epsilon() {
        // s0 --ε--> s1 --ε--> s2 --ε--> *s3*
        let mut a = NFA::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();
        a.set_initial(s0).unwrap();
        a.add_final(s3).unwrap();
        a.add_transition(s0, s1, TransitionType::Epsilon).unwrap();
        a.add_transition(s1, s2, TransitionType::Epsilon).unwrap();
        a.add_transition(s2, s3, TransitionType::Epsilon).unwrap();
        let result = a.eliminate_epsilon();
        assert_eq!(result.states.len(), 1);
        assert_eq!(result.initial, Some(0));
        assert_eq!(result.finals.len(), 1);
    }

    #[test]
    fn test_elim_epsilon_with_direct_transition() {
        // s0 --(a|ε)--> s1
        let mut a = NFA::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        a.set_initial(s0).unwrap();
        a.add_final(s1).unwrap();
        a.add_transition(s0, s1, TransitionType::Range(CharRange::singleton('a')))
            .unwrap();
        a.add_transition(s0, s1, TransitionType::Epsilon).unwrap();
        let result = a.eliminate_epsilon();
        assert_eq!(result.states.len(), 2);
        assert_eq!(result.initial, Some(0));
        assert_eq!(result.finals.len(), 2);
    }

    /* Tests for trimming */

    #[test]
    fn test_trim_empty() {
        let a = NFA::new();
        let result = a.trim();
        assert_eq!(result.states.len(), 0);
        assert_eq!(result.initial, None);
        assert_eq!(result.finals.len(), 0);
    }

    #[test]
    fn test_trim_no_finals() {
        let mut a = NFA::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        a.set_initial(s0).unwrap();
        a.add_transition(s0, s1, TransitionType::Epsilon).unwrap();
        let result = a.trim();
        assert_eq!(result.states.len(), 0);
        assert_eq!(result.initial, None);
        assert_eq!(result.finals.len(), 0);
    }

    #[test]
    fn test_trim_no_initial() {
        let mut a = NFA::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        a.add_final(s1).unwrap();
        a.add_transition(s0, s1, TransitionType::Epsilon).unwrap();
        let result = a.trim();
        assert_eq!(result.states.len(), 0);
        assert_eq!(result.initial, None);
        assert_eq!(result.finals.len(), 0);
    }

    #[test]
    fn test_trim_single_state_automaton() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        nfa.add_final(q0).unwrap(); // The only state is both initial and final

        let trimmed = nfa.trim();

        assert!(trimmed.accepts(&"".into())); // Should accept the empty string
        assert_eq!(trimmed.num_states(), 1);
    }

    #[test]
    fn test_trim_disconnected_states() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();
        let q3 = nfa.new_state(); // Completely disconnected state

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();
        nfa.add_final(q3).unwrap(); // Should be removed because it's unreachable

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('x', 'x')))
            .unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::new('y', 'y')))
            .unwrap();

        let trimmed = nfa.trim();

        assert!(trimmed.accepts(&"xy".into()));
        assert_eq!(trimmed.num_states(), 3);
    }

    #[test]
    fn test_trim_dead_states() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();
        let q3 = nfa.new_state(); // This state has no path to final states

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::new('b', 'b')))
            .unwrap();
        nfa.add_transition(q3, q3, TransitionType::Range(CharRange::new('c', 'c')))
            .unwrap(); // Useless state

        let trimmed = nfa.trim();

        assert!(trimmed.accepts(&"ab".into()));
        assert_eq!(trimmed.num_states(), 3); // Dead-end state removed
    }

    #[test]
    fn test_trim_already_trimmed() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::new('b', 'b')))
            .unwrap();

        let trimmed = nfa.trim();

        assert_eq!(nfa.num_states(), trimmed.num_states()); // Should be the same size
        assert!(trimmed.accepts(&"ab".into()));
    }
}
