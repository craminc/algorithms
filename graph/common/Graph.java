package graph.common;

import java.util.*;
import java.util.stream.Collectors;

public class Graph<T> {

    private final Map<T, Node<T>> map = new HashMap<>();

    public static class Node<T> {
        private final T val;
        private final List<Node<T>> successor;

        public Node(T val) {
            this.val = val;
            this.successor = new ArrayList<>();
        }
    }

    public Graph<T> addEdge(T cur, T nxt) {
        Node<T> curNode = map.get(cur);
        if (curNode == null) {
            curNode = new Node<>(cur);
            map.put(cur, curNode);
        }
        Node<T> nxtNode = map.get(nxt);
        if (nxtNode == null) {
            nxtNode = new Node<>(nxt);
            map.put(nxt, nxtNode);
        }
        if (!curNode.successor.contains(nxtNode))
            curNode.successor.add(nxtNode);
        return this;
    }

    public List<T> successors(T cur) {
        Node<T> curNode = map.get(cur);
        if (curNode == null) return Collections.emptyList();
        return curNode.successor.stream().map(t -> t.val).collect(Collectors.toList());
    }

    public List<T> nodes() {
        return new ArrayList<>(map.keySet());
    }
}
