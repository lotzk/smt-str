//! Module for dealing with finite automata.

use std::{
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    error::Error,
    fmt::{Display, Formatter},
};

pub mod comp;
mod compile;
pub mod det;
mod dfa;
mod inter;
mod nfa;
use compile::Thompson;
pub use dfa::{DState, DFA};
pub use nfa::{NState, NFA};

use crate::{
    alphabet::CharRange,
    re::{ReBuilder, Regex},
    SmtChar, SmtString,
};

extern crate dot as dotlib;

mod dot;

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum AutomatonError {
    StateNotFound(StateId),
    /// The automaton is not epsilon-free, but the operation requires an epsilon-free automaton
    RequiresEpsilonFree(String),
    /// Returned if the automaton is expected to be deterministic, but is not
    /// In theory, this should never happen, as the dfa should be deterministic by construction
    NotDeterministic,
}
impl Error for AutomatonError {}

pub fn compile(re: &Regex, builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
    let mut compiler = Thompson::default();
    compiler.compile(re, builder)
}

/// Identifier for a state that is unique within an automaton
pub type StateId = usize;

/// The type of a transition in an automaton.
/// A transition can be one of the following:
/// - A range of characters, which matches all characters in the given range. If the next input character is in the range, the automaton transitions to the destination state. The range can contain a single character, in which case the transition is equivalent to a character transition. The range can also be the full unicode range, which matches any character. Lastly, the range can be empty, which models a transition that is not feasible, i.e., never taken.
/// - An epsilon transition, which does not consume any input and transitions to the destination state.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Copy)]
pub enum TransitionType {
    Range(CharRange),
    NotRange(CharRange),
    Epsilon,
}

impl TransitionType {
    /// Creates a character transition that matches the given character.
    pub fn char(c: impl Into<SmtChar>) -> Self {
        TransitionType::Range(CharRange::singleton(c.into()))
    }

    /// Creates a range transition that matches all characters in the given range [l, u] (both inclusive).
    /// This is equivalent to using `TransitionType::Range(CharRange::new(l, u))`.
    pub fn range(l: impl Into<SmtChar>, u: impl Into<SmtChar>) -> Self {
        TransitionType::Range(CharRange::new(l, u))
    }

    /// Creates a transition that matches any character.
    pub fn is_epsilon(&self) -> bool {
        matches!(self, TransitionType::Epsilon)
    }
}
impl Display for TransitionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TransitionType::Range(r) => write!(f, "{}", r),
            TransitionType::NotRange(r) => write!(f, "¬{}", r),
            TransitionType::Epsilon => write!(f, "ε"),
        }
    }
}

/// A transition in an automaton, which consists of a type and a destination state.
/// The type of a transition determines which input symbols it matches.
/// The destination state is the state that the automaton transitions to when the transition is taken.
/// The destination state is identified by its unique identifier within the automaton.
/// If a transition leads to a state that does not exist in the automaton, the automaton is considered to be in an error state.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Transition {
    type_: TransitionType,
    destination: StateId,
}

impl Transition {
    /// Creates a new transition with the given type and destination state.
    pub fn new(ttype: TransitionType, destination: StateId) -> Self {
        Self {
            type_: ttype,
            destination,
        }
    }

    /// Creates a new transition that matches the given character and transitions to the given destination state.
    pub fn char(c: impl Into<SmtChar>, destination: StateId) -> Self {
        Self::new(TransitionType::char(c), destination)
    }

    /// Creates a new transition that matches any character and transitions to the given destination state.
    pub fn any(destination: StateId) -> Self {
        Self::new(TransitionType::Range(CharRange::all()), destination)
    }

    /// Creates a new transition that matches all characters in the given range and transitions to the given destination state.
    pub fn range_from(
        start: impl Into<SmtChar>,
        end: impl Into<SmtChar>,
        destination: StateId,
    ) -> Self {
        Self::range(CharRange::new(start, end), destination)
    }

    /// Creates a new transition that matches all characters in the given range and transitions to the given destination state.
    pub fn range(r: CharRange, destination: StateId) -> Self {
        Self::new(TransitionType::Range(r), destination)
    }

    /// Creates a new transition that matches all characters in the given range and transitions to the given destination state.
    pub fn not_range(r: CharRange, destination: StateId) -> Self {
        Self::new(TransitionType::NotRange(r), destination)
    }

    /// Creates a new epsilon transition that transitions to the given destination state.
    pub fn epsilon(destination: StateId) -> Self {
        Self {
            type_: TransitionType::Epsilon,
            destination,
        }
    }

    /// Returns the destination state of this transition.
    pub fn get_dest(&self) -> StateId {
        self.destination
    }

    /// Returns `true` if this transition is an epsilon transition
    fn is_epsilon(&self) -> bool {
        matches!(self.type_, TransitionType::Epsilon)
    }

    /// Returns `true` if the given symbol matches this transition
    ///
    /// # Example
    /// ```
    /// use smt_strings::automata::Transition;
    ///
    /// // Single character transition
    /// let t = Transition::char('a', 1);
    /// assert!(t.matches('a'));
    /// assert!(!t.matches('b'));
    ///
    /// // Range transition
    /// let r = Transition::range_from('a', 'z', 1);
    /// assert!(r.matches('a'));
    /// assert!(r.matches('d'));
    /// assert!(!r.matches('0'));
    ///
    /// // Epsilon transition, does not match any character
    /// let e = Transition::epsilon(1);
    /// assert!(!e.matches('a'));
    /// ```
    pub fn matches(&self, c: impl Into<SmtChar>) -> bool {
        match self.type_ {
            TransitionType::Range(r) => r.contains(c),
            TransitionType::NotRange(r) => !r.contains(c),
            TransitionType::Epsilon => false,
        }
    }

    /// Returns the [TransitionType] of this transition.
    pub fn get_type(&self) -> &TransitionType {
        &self.type_
    }

    /// Returns `true` if this transition is feasible, i.e., if it can be taken.
    /// A transition is feasible it is a non-empty range transition or an epsilon transition.
    pub fn feasible(&self) -> bool {
        match self.type_ {
            TransitionType::Range(r) => !r.is_empty(),
            TransitionType::NotRange(r) => r.complement().iter().any(|r| !r.is_empty()),
            TransitionType::Epsilon => true,
        }
    }

    /// Returns a new transition with the same type as this transition, but a different destination state.
    ///
    /// # Example
    /// ```
    /// use smt_strings::automata::Transition;
    /// let t = Transition::char('a', 1);
    /// let t2 = t.new_dest(2);
    /// assert_eq!(t2.get_dest(), 2);
    /// assert_eq!(t2.get_type(), t.get_type());
    /// ```
    pub fn new_dest(&self, dest: StateId) -> Self {
        Self {
            type_: self.type_,
            destination: dest,
        }
    }
}

/// A state in an automaton.
/// A state is merely a vertex in the transition graph of the automaton.
/// It consists of a set of transitions to other states.
/// Each state in an automaton is identified identifier that is unique within the automaton.
/// This identifier is used to refer to the destination state in transitions.
///
/// This trait is implement for both deterministic and non-deterministic automata.
/// See the [DFA] and [NFA] types for more information.
pub trait State: Clone + Default {
    /// Returns the transitions leaving this state
    fn transitions(&self) -> Vec<Transition>;
    /// Adds a new transition to this state
    fn add_transition(&mut self, transition: Transition);
    /// Removes the given transition from this state
    fn remove_transition(&mut self, transition: &Transition);
    /// Returns a new state with all destinations offset by the given amount.
    /// Meaning, if the destination of a transition is `d`, the new destination is `d + offset`.
    fn offset_transitions(&self, offset: usize) -> Self;
}

/// A finite automaton, which consists of a set of states, an initial state, a set of final states.
/// Every state is identified by a unique identifier and contains a set of [Transition]s to other states.
/// Every transition defines the input symbols that can be consumed and the destination state it transitions to.
/// The destination state is identified by its identifier.
///
/// The type of [State] is generic:
/// - For deterministic automata, the state is a [DState], which contains a single transition for each input symbol.
/// - For non-deterministic automata, the state is a [NState], which contains a (possibly-empty) set of transitions for each input symbol.
#[derive(Debug, Clone, Default)]
pub struct Automaton<S: State> {
    /// The states of this automaton
    states: Vec<S>,
    /// The initial state of this automaton
    initial: Option<StateId>,
    /// The set of final states of this automaton
    finals: HashSet<StateId>,
    /// Whether this automaton is trim, i.e., every state is reachable from the initial state and every state can reach a final state
    /// This flag is set to `false` whenever the automaton is modified, and set to `true` after calling the `trim` method.
    /// This flag is used to avoid unnecessary trimming of the automaton, when it is already trim.
    trim: bool,
}

impl<S: State> Automaton<S> {
    /// Create a new, empty automaton, i.e., it does neither have any states.
    pub fn new() -> Self {
        Self {
            states: vec![],
            initial: None,
            finals: HashSet::new(),
            trim: true,
        }
    }

    /// Create a new automaton with the given states.
    /// The initial and final states are not set and must be set manually using the `set_initial` and `add_final` methods.
    pub fn from_states(states: Vec<S>) -> Self {
        Self {
            states,
            initial: None,
            finals: HashSet::new(),
            trim: false,
        }
    }

    /// Adds an existing state to this automaton and returns its identifier.
    pub fn add_state(&mut self, state: S) -> StateId {
        self.states.push(state);
        self.trim = false;
        self.states.len() - 1
    }

    /// Creates a new state and adds it to this automaton.
    /// Returns the identifier of the new state.
    pub fn new_state(&mut self) -> StateId {
        self.add_state(S::default())
    }

    /// Sets the given state as the initial state of this automaton.
    /// Returns the previous initial state, if any.
    pub fn set_initial(&mut self, state: StateId) -> Option<StateId> {
        self.trim = false;
        self.initial.replace(state)
    }

    /// Adds a transition from the source state to the destination state with the given type.
    /// Both the source and destination states must exist in this automaton.
    ///
    /// # Returns
    /// If either state does not exist, an [AutomatonError::StateNotFound] error is returned.
    /// Returns `Ok(())` if the transition was added successfully.
    pub fn add_transition(
        &mut self,
        src: StateId,
        dest: StateId,
        ttype: TransitionType,
    ) -> Result<(), AutomatonError> {
        let qsrc = self.get_state_mut(src)?;
        qsrc.add_transition(Transition {
            type_: ttype,
            destination: dest,
        });
        Ok(())
    }

    /// Removes the given transition from the source state.
    ///
    /// # Returns
    /// If the source state does not exist, an [AutomatonError::StateNotFound] error is returned.
    /// Returns `Ok(())` if the transition was removed successfully or did not exist.
    pub fn remove_transition(
        &mut self,
        src: StateId,
        transition: &Transition,
    ) -> Result<(), AutomatonError> {
        let qsrc = self.get_state_mut(src)?;
        qsrc.remove_transition(transition);
        self.trim = false;
        Ok(())
    }

    /// Adds the given state to the set of final states of this automaton.
    /// Does not check if the state exists in this automaton.
    pub fn add_final(&mut self, state: StateId) {
        self.finals.insert(state);
    }

    /// Returns a vector of all state Id in this automaton.
    /// The length of the vector is equal to the number of states in this automaton.
    pub fn states(&self) -> Vec<StateId> {
        (0..self.states.len()).collect()
    }

    /// Returns the number of states in this automaton
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Returns the number of transitions in this automaton.
    /// This requires a linear search through all states and transitions.
    pub fn num_transitions(&self) -> usize {
        self.states.iter().map(|s| s.transitions().len()).sum()
    }

    /// Returns an iterator over the states of this automaton.
    pub fn iter_states(&self) -> impl Iterator<Item = (StateId, &S)> {
        self.states.iter().enumerate()
    }

    /// Returns an iterator over the states of this automaton, allowing mutable access to the states.
    /// Calling this method marks the automaton as untrimmed, because the mutable references can be used to alter the automaton.
    pub fn iter_states_mut(&mut self) -> impl Iterator<Item = (StateId, &mut S)> {
        self.trim = false;
        self.states.iter_mut().enumerate()
    }

    /// Returns the state with the given identifier.
    pub fn get_state(&self, id: StateId) -> Result<&S, AutomatonError> {
        self.states.get(id).ok_or(AutomatonError::StateNotFound(id))
    }

    /// Returns a mutable reference to the state with the given identifier.
    /// Calling this method marks the automaton as untrimmed, because the mutable reference can be used to alter the automaton.
    pub fn get_state_mut(&mut self, id: StateId) -> Result<&mut S, AutomatonError> {
        // Mark the automaton as untrimmed because the mutable reference can be used to remove transitions
        self.trim = false;
        self.states
            .get_mut(id)
            .ok_or(AutomatonError::StateNotFound(id))
    }

    /// Returns the initial state of this automaton
    pub fn initial(&self) -> Option<StateId> {
        self.initial
    }

    /// Returns `true` if the given state is the initial state of this automaton
    pub fn is_initial(&self, state: StateId) -> bool {
        self.initial == Some(state)
    }

    /// Returns the set of final states of this automaton
    pub fn finals(&self) -> &HashSet<StateId> {
        &self.finals
    }

    /// Clears the set of final states of this automaton.
    /// This does not remove the states from the automaton.
    /// After calling this method, the automaton is marked as untrimmed.
    pub fn clear_finals(&mut self) {
        self.trim = false;
        self.finals.clear();
    }

    /// Returns `true` if the given state is a final state of this automaton
    pub fn is_final(&self, state: StateId) -> bool {
        self.finals.contains(&state)
    }

    /// Returns the successors of the given state
    /// Returns an empty vector if the given state does not exist in this automaton.
    pub fn successors(&self, state: StateId) -> Vec<StateId> {
        self.states
            .get(state)
            .map(|s| s.transitions().into_iter().map(|t| t.get_dest()).collect())
            .unwrap_or_default()
    }

    /// Returns the predecessors of the given state
    /// This requires a linear search through all states and transitions.
    pub fn predecessors(&self, state: StateId) -> Vec<StateId> {
        let mut preds = vec![];
        for (i, s) in self.states.iter().enumerate() {
            for t in s.transitions().iter() {
                if t.get_dest() == state {
                    preds.push(i);
                }
            }
        }
        preds
    }

    /// Returns a map from state ids to the set of states that can reach the state with the given id.
    /// This is preferred over calling `predecessors` for each state, because it only requires a single pass through the automaton.
    /// However, the returned map becomes invalid if the automaton is modified.
    ///
    /// # Returns
    /// A map from state ids to the set of states that can reach the state with the given id.
    /// For example, if the automaton has a transition from state 0 to state 1 and from state 1 to state 1, the map will contain the entry `(1, {0, 1})`.
    /// If the automaton is trim, then each state except the initial state has at least one predecessor.
    /// For states without predecessors, the map has no entry for that state. Meaning, if state 0 has no predecessors, the map does not contain the key 0.
    fn predecessor_map(&self) -> HashMap<StateId, HashSet<StateId>> {
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        let mut preds = HashMap::new();
        if let Some(q0) = self.initial {
            queue.push_back(q0);
            seen.insert(q0);
        }
        while let Some(q) = queue.pop_front() {
            for t in self.get_state(q).unwrap().transitions() {
                let dest = t.get_dest();
                if !seen.contains(&dest) {
                    queue.push_back(dest);
                    seen.insert(dest);
                }
                preds.entry(dest).or_insert(HashSet::new()).insert(q);
            }
        }
        preds
    }

    /// Returns the epsilon closure of the given state.
    /// The epsilon closure of a state is the set of states that can be reached from the given state by following epsilon transitions, i.e., by not consuming any input.
    /// For deterministic automata, which does not allow epsilon transitions, this is always a singleton set containing the given state.
    pub fn epsilon_closure(&self, state: StateId) -> Result<HashSet<StateId>, AutomatonError> {
        let mut closure = HashSet::new();
        let mut stack = vec![state];
        while let Some(q) = stack.pop() {
            closure.insert(q);
            for t in self.get_state(q)?.transitions().iter() {
                if t.is_epsilon() {
                    let dest = t.get_dest();
                    if !closure.contains(&dest) {
                        stack.push(dest);
                    }
                }
            }
        }
        Ok(closure)
    }

    /// Returns the automaton that accept the intersection of the languages of this and the other automaton.
    /// The intersection of two languages is the set of words that are accepted by both automatons.
    /// The automaton that accepts the intersection is computed using the product construction, which takes O(nm) time, where n and m are the number of states in the two automata.
    /// For deterministic automatons, the intersection is also deterministic. For non-deterministic automatons, the intersection is non-deterministic.
    pub fn intersect(&self, other: &Self) -> Result<Self, AutomatonError> {
        inter::intersect(self, other)
    }

    /// Returns true iff this automaton is acyclic.
    /// If the automaton is acyclic, then it accepts a finite language.
    /// The converse only holds when the automaton is epsilon-free because a cycle on epsilon transitions can be traversed infinitely often without consuming any input.
    /// Thus, if the automaton is epsilon-free, then it accepts a finite language iff it is acyclic.
    pub fn acyclic(&self) -> Result<bool, AutomatonError> {
        fn has_cycle<S: State>(
            nfa: &Automaton<S>,
            path: Vec<usize>,
            s: usize,
        ) -> Result<bool, AutomatonError> {
            let q = nfa.get_state(s)?;
            for t in q.transitions() {
                let mut path = path.clone();
                path.push(s);
                if path.contains(&t.get_dest()) || has_cycle(nfa, path, t.get_dest())? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        if let Some(q0) = self.initial {
            has_cycle(self, vec![], q0).map(|b| !b)
        } else {
            Ok(true)
        }
    }

    /// Returns the shortest path from the initial state to a final state, or `None` if there is no such path.
    /// The length of the path is the number of transitions on the path.
    /// This corresponds to the shortest word in the language of the automaton.
    pub fn shortest(&self) -> Option<usize> {
        // Do a BFS from the initial state to find the shortest path to a final state
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        queue.push_back((self.initial?, 0));
        seen.insert(self.initial?);
        while let Some((q, len)) = queue.pop_front() {
            if self.is_final(q) {
                return Some(len);
            }
            let state = self.get_state(q).unwrap();
            for t in state.transitions() {
                let dest = t.get_dest();
                if !seen.contains(&dest) {
                    seen.insert(dest);
                    if t.is_epsilon() {
                        queue.push_front((dest, len));
                    } else {
                        queue.push_back((dest, len + 1));
                    }
                }
            }
        }
        None
    }

    /// Returns the longest acyclic path from the initial state to a final state, or `None` if there is no such path.
    /// This corresponds to the depth in the tree where the initial state is the root and the final states are the leaves.
    /// Panics on epsilon transitions.
    pub fn longest(&self) -> Option<usize> {
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        queue.push_back((self.initial?, 0));
        seen.insert(self.initial?);

        let mut longest = -1;
        while let Some((q, len)) = queue.pop_front() {
            let state = self.get_state(q).unwrap();
            for t in state.transitions() {
                if t.is_epsilon() {
                    panic!("Epsilon transitions are not allowed");
                }
                let dest = t.get_dest();
                if !seen.contains(&dest) {
                    seen.insert(dest);
                    queue.push_back((dest, len + 1));
                    if self.is_final(dest) {
                        longest = longest.max(len + 1);
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

    /// Samples a word from the language of this automaton.
    /// The word is generated by following transitions from the initial state to a final state and recording the input symbols.
    /// The methods avoids cycles by keeping track of the states that have been visited. Thus, any cycle in the automaton is not part of the generated word.
    ///
    /// The following guarantees are provided:
    /// - If the automaton is trim and non-empty, this method always returns a word.
    /// - If the automaton is empty, this method returns `None`.
    /// - If the automaton is not trim it is not guaranteed that this method returns a word, even if the automaton is non-empty.
    #[cfg(feature = "sampling")]
    pub fn sample(&self) -> Option<SmtString> {
        let mut word = SmtString::empty();
        let mut current = self.initial?;
        let mut seen = HashSet::new();
        seen.insert(current);
        while !self.is_final(current) {
            let mut next = None;
            for t in self.get_state(current).unwrap().transitions() {
                if t.feasible() && !seen.contains(&t.get_dest()) {
                    seen.insert(t.get_dest());
                    next = Some(t);
                    break;
                }
            }
            let t = next?;
            match t.get_type() {
                TransitionType::Range(r) => {
                    word.push(r.start());
                }
                TransitionType::NotRange(r) => {
                    if let Some(r) = r.complement().first() {
                        word.push(r.start())
                    } else {
                        return None;
                    }
                }
                TransitionType::Epsilon => (),
            }
            current = t.get_dest();
        }
        Some(word)
    }

    /// Returns the set of states that can reach a final state.
    fn useful_states(&self) -> HashSet<StateId> {
        let mut useful = HashSet::new();
        let mut queue = VecDeque::new();
        // Find all states that can reach a final state by doing a BFS from the final states on the reverse automaton
        let preds = self.predecessor_map();
        for &qf in &self.finals {
            queue.push_back(qf);
            useful.insert(qf);
        }
        while let Some(q) = queue.pop_front() {
            if let Some(p) = preds.get(&q) {
                for &pred in p {
                    if useful.insert(pred) {
                        queue.push_back(pred);
                    }
                }
            }
        }
        useful
    }

    /// Trims the automaton. An automaton is called *trim* if
    ///
    /// - every state is reachable from the initial state and
    /// - every state can reach a final state
    ///
    /// In that case, *sink* states are implicit: The automaton rejects a word if it cannot make a transition from the current state on the next input symbol.
    /// Calling this method has no effect if the automaton is already trim.
    /// Specifically, if the automaton is not modified after calling this method, any subsequent call is a no-op and terminates immediately.
    pub fn trim(&mut self) {
        if self.trim {
            return;
        }
        // Maps state ids in this automaton to the corresponding state ids in the trimmed automaton
        let mut state_map = HashMap::new();

        // The trimmed automaton
        let mut trim_states = vec![];
        let mut initial = None;
        let mut finals = HashSet::new();
        // The transitions in the trimmed automaton
        let mut transitions: Vec<(StateId, Transition)> = vec![];

        // Only consider states that can reach a final state
        let useful = self.useful_states();

        // Do a BFS from the initial state to find all reachable states that are useful
        let mut queue = VecDeque::new();
        if let Some(q0) = self.initial {
            if useful.contains(&q0) {
                queue.push_back(q0);
                state_map.insert(q0, 0); // Initial state gets id 0 in the new automaton
                initial = Some(0);
                trim_states.push(S::default());
            }
        }
        while let Some(q) = queue.pop_front() {
            // Check if the current state is final
            if self.finals.contains(&q) {
                finals.insert(state_map[&q]);
            }
            // Add all successors to the queue, if they are useful
            for t in self.get_state(q).unwrap().transitions().iter() {
                let dest = t.get_dest();
                if useful.contains(&dest) {
                    let trim_dest_id = match state_map.entry(dest) {
                        Entry::Occupied(l) => *l.into_mut(),
                        Entry::Vacant(v) => {
                            trim_states.push(S::default());
                            let id = trim_states.len() - 1;
                            // add to queue, need to process this state
                            queue.push_back(dest);
                            v.insert(id);
                            id
                        }
                    };
                    transitions.push((state_map[&q], t.new_dest(trim_dest_id)));
                }
            }
        }
        // Add all transitions to the trimmed automaton
        for (src, t) in transitions {
            trim_states[src].add_transition(t);
        }

        // Replace the old automaton with the trimmed automaton
        self.states = trim_states;
        self.initial = initial;
        self.finals = finals;
        self.trim = true;
    }

    /// Returns a string representation of this automaton in the DOT format.
    pub fn dot(&self) -> Option<String> {
        let mut buf = Vec::new();
        dotlib::render(self, &mut buf).unwrap();
        String::from_utf8(buf).ok()
    }

    /// Returns `true` if this automaton accepts no words, i.e., if its language is empty.
    /// Returns `false` if this automaton accepts at least one word.
    pub fn is_empty(&self) -> bool {
        if self.trim {
            // If the automaton is trim, it is empty if it has no initial state or no final states
            self.initial.is_none() || self.finals.is_empty()
        } else {
            self.initial.is_none() || self.finals.is_empty() || self.shortest().is_none()
        }
    }

    /// Applies the transition function to the given state by consuming the input symbol.
    /// Meaning, assumes the automaton is in the given state and reads the given input symbol.
    ///
    /// # Returns
    /// If the given state does not exist, an [AutomatonError::StateNotFound] error is returned.
    /// Returns the set of states that the automaton can reach from the given state by reading the given input symbol.
    /// For deterministic automata, this set contains at most one state.
    /// If the returned set is empty, the automaton rejects the input symbol in the given state.
    pub fn consume(&self, q: StateId, c: SmtChar) -> Result<HashSet<StateId>, AutomatonError> {
        let mut next = HashSet::new();
        for t in self.get_state(q)?.transitions().iter() {
            if t.matches(c) {
                next.insert(t.get_dest());
            }
        }
        Ok(next)
    }

    /// Returns the set of states reachable from the initial state by reading the given word
    fn run(&self, word: &SmtString) -> Result<HashSet<StateId>, AutomatonError> {
        if self.initial.is_none() {
            return Ok(HashSet::new());
        }
        let mut current = self.epsilon_closure(self.initial().unwrap())?;
        for c in word.iter() {
            let mut next = HashSet::new();
            for state in current.iter().filter_map(|id| self.get_state(*id).ok()) {
                for transition in state.transitions().iter() {
                    if transition.matches(*c) {
                        next.extend(self.epsilon_closure(transition.get_dest())?);
                    }
                }
            }
            current = next;
        }
        Ok(current)
    }

    /// Returns `true` if and only if the given word is accepted by this automaton
    pub fn accepts(&self, word: &SmtString) -> Result<bool, AutomatonError> {
        let reached = self.run(word)?;
        Ok(reached.intersection(&self.finals).count() > 0)
    }
}

impl Display for Transition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.type_ {
            TransitionType::Range(r) => {
                write!(f, "({} -> {})", r, self.destination)
            }
            TransitionType::NotRange(r) => {
                write!(f, "(¬{} -> {})", r, self.destination)
            }
            TransitionType::Epsilon => write!(f, "(ε -> {})", self.destination),
        }
    }
}

impl Display for AutomatonError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AutomatonError::StateNotFound(s) => write!(f, "State {} not found", s),
            AutomatonError::RequiresEpsilonFree(s) => write!(f, "Requires epsilon freeness: {}", s),
            AutomatonError::NotDeterministic => write!(f, "DFA is not deterministic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_dfa_unreachable_state() {
        let mut dfa = DFA::new();

        // Adding states
        let q0 = dfa.new_state(); // Initial state
        let q1 = dfa.new_state(); // Final state
        let _ = dfa.new_state(); // Unreachable state

        // Mark initial and final states
        dfa.set_initial(q0);
        dfa.add_final(q1);

        // Adding transitions
        dfa.add_transition(q0, q1, TransitionType::char('a'))
            .unwrap();

        // Applying trim
        dfa.trim();

        // Expected results
        assert_eq!(dfa.states.len(), 2); // Only q0 and q1 should remain
        assert_eq!(dfa.initial, Some(0)); // Initial state should be 0
        assert!(dfa.finals.contains(&1)); // Final state should be 1
    }

    #[test]
    fn test_trim_dfa_with_unreachable_final_state() {
        let mut dfa = DFA::new();

        // Adding states
        let q0 = dfa.new_state(); // Initial state
        let q1 = dfa.new_state(); // Final state
        let q2 = dfa.new_state(); // Another state

        let q3 = dfa.new_state(); // Unreachable final state

        // Mark initial and final states
        dfa.set_initial(q0);
        dfa.add_final(q2);

        dfa.add_final(q3); // This final state is unreachable

        // Adding transitions
        dfa.add_transition(q0, q1, TransitionType::char('a'))
            .unwrap();
        dfa.add_transition(q1, q2, TransitionType::char('b'))
            .unwrap();

        // Applying trim
        dfa.trim();

        // Expected results
        println!("{}", dfa.dot().unwrap());
        assert_eq!(dfa.states.len(), 3); // Only q0, q1, and q2 should remain
        assert_eq!(dfa.initial, Some(0)); // Initial state should be 0
        assert!(dfa.accepts(&"ab".into()).unwrap()); // The automaton should still accept "ba"
    }

    #[test]
    fn test_trim_dfa_useless_state() {
        let mut dfa = DFA::new();

        // Adding states
        let q0 = dfa.new_state(); // Initial state
        let q1 = dfa.new_state(); // Final state
        let q2 = dfa.new_state(); // Another state
        let q3 = dfa.new_state(); // Unreachable final state

        // Mark initial and final states
        dfa.set_initial(q0);
        dfa.add_final(q2);

        // Adding transitions
        dfa.add_transition(q0, q1, TransitionType::char('a'))
            .unwrap();
        dfa.add_transition(q1, q2, TransitionType::char('b'))
            .unwrap();
        // q3 cannot reach any final state
        dfa.add_transition(q0, q3, TransitionType::char('c'))
            .unwrap();

        // Applying trim
        dfa.trim();

        // Expected results
        assert_eq!(dfa.states.len(), 3); // Only q0, q1, and q2 should remain
        assert_eq!(dfa.initial, Some(0)); // Initial state should be 0
        assert!(dfa.finals.contains(&2)); // Final state should be 2
        assert!(!dfa.finals.contains(&3)); // q3 should be removed
    }
}
