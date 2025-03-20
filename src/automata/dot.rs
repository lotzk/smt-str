//! Facilities to generate a DOT representation of an NFA.

use super::{StateId, Transition, TransitionType, NFA};

impl<'a> dot::Labeller<'a, StateId, (StateId, Transition, StateId)> for NFA {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("automaton").unwrap()
    }

    fn node_id(&'a self, n: &StateId) -> dot::Id<'a> {
        dot::Id::new(format!("q{}", n)).unwrap()
    }

    fn node_shape(&'a self, node: &StateId) -> Option<dot::LabelText<'a>> {
        if self.finals.contains(node) {
            return Some(dot::LabelText::LabelStr("doublecircle".into()));
        }

        None
    }

    fn node_label(&'a self, n: &StateId) -> dot::LabelText<'a> {
        if let Some(q0) = self.initial {
            if q0 == *n {
                return dot::LabelText::LabelStr(
                    format!("{} (Init)", self.node_id(n).name()).into(),
                );
            }
        }
        dot::LabelText::LabelStr(self.node_id(n).name())
    }

    fn edge_label(&'a self, e: &(StateId, Transition, StateId)) -> dot::LabelText<'a> {
        match e.1.get_type() {
            TransitionType::Range(r) => dot::LabelText::LabelStr(format!("{}", r).into()),
            TransitionType::NotRange(r) => dot::LabelText::LabelStr(format!("not({})", r).into()),
            TransitionType::Epsilon => dot::LabelText::LabelStr("".into()),
        }
    }

    fn node_style(&'a self, _n: &StateId) -> dot::Style {
        dot::Style::None
    }

    fn node_color(&'a self, _node: &StateId) -> Option<dot::LabelText<'a>> {
        None
    }

    fn kind(&self) -> dot::Kind {
        dot::Kind::Digraph
    }
}

impl<'a> dot::GraphWalk<'a, StateId, (StateId, Transition, StateId)> for NFA {
    fn nodes(&'a self) -> dot::Nodes<'a, StateId> {
        self.states
            .iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
            .into()
    }

    fn edges(&'a self) -> dot::Edges<'a, (StateId, Transition, StateId)> {
        let mut edges: Vec<(StateId, Transition, StateId)> = vec![];
        for (i, state) in self.states.iter().enumerate() {
            for transition in state.transitions() {
                edges.push((i, transition.clone(), transition.get_dest()));
            }
        }
        edges.into()
    }

    fn source(&'a self, edge: &(StateId, Transition, StateId)) -> StateId {
        edge.0
    }

    fn target(&'a self, edge: &(StateId, Transition, StateId)) -> StateId {
        edge.2
    }
}
