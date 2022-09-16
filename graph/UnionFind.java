package graph;

public class UnionFind {

    // count record the num of connected components
    private int count;
    // parent[i] record the parent of node i
    private final int[] parent;
    // record the scale of connected component
    private final int[] size;

    public UnionFind(int n) {
        this.count = n;
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    /* union two node */
    public void union(int p, int q) {
        int pp = find(p);
        int qp = find(q);
        if (pp == qp) return;
        if (size[pp] < size[qp]) {
            parent[pp] = qp;
            size[qp] += size[pp];
        } else {
            parent[qp] = pp;
            size[pp] += size[qp];
        }
        // delete one connected component
        count--;
    }

    /* check if two nodes are connected */
    public boolean connected(int p, int q) {
        return find(p) == find(q);
    }

    /* get the num of connected components */
    public int count() {
        return count;
    }

    /* get the root of specify node (use path compression)*/
    private int find(int x) {
        // compress depth to 2
        if (x != parent[x]){
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

}
