package graph;

import graph.common.Graph;

import java.util.*;

public class Tarjan {

    private final Graph<Integer> graph;
    private final Deque<Integer> stack;
    /**
     * strong connected component
     */
    private final List<List<Integer>> sccs = new ArrayList<>();
    /**
     * the step
     */
    private final int[] dfn;
    private final int[] low;
    /**
     * record the step for traverse
     */
    private int step = 0;

    public Tarjan(Graph<Integer> graph) {
        this.graph = graph;
        this.stack = new ArrayDeque<>();
        int n = graph.nodes().size();
        dfn = new int[n];
        low = new int[n];
        // init dfn low, -1 means not visit
        Arrays.fill(dfn, -1);
        Arrays.fill(low, -1);
    }

    public List<List<Integer>> findSccs() {
        for (int i = 0; i < graph.nodes().size(); i++) {
            if (dfn[i] == -1) dfs(i);
        }
        return sccs;
    }

    public void dfs(int cur) {
        dfn[cur] = low[cur] = ++step;
        stack.push(cur);

        for (int nxt : graph.successors(cur)) {
            // dfs successors
            if (dfn[nxt] == -1)
                dfs(nxt);
            // update low
            if (stack.contains(nxt))
                low[cur] = Math.min(low[cur], low[nxt]);

        }

        List<Integer> scc = new ArrayList<>();
        if (dfn[cur] == low[cur]) {
            // find the main element of scc
            int j;
            do {
                // pop all element of the scc
                j = stack.pop();
                scc.add(j);
            } while (cur != j);
            sccs.add(scc);
        }
    }

    public static void main(String[] args) {
        Graph<Integer> g = new Graph<Integer>()
                .addEdge(0, 1)
                .addEdge(1, 3)
                .addEdge(0, 2)
                .addEdge(2, 3)
                .addEdge(3, 0)
                .addEdge(2, 4)
                .addEdge(4, 5)
                .addEdge(3, 5);
        Tarjan tarjan = new Tarjan(g);
        List<List<Integer>> sccs = tarjan.findSccs();
        System.out.println(sccs);
    }
}
