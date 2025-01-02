package graph;

public class Dijkstra {
    private final int[][] graph = new int[][]{
            { 0, 10, -1, -1,  5},
            {-1,  0,  1, -1,  2},
            {-1, -1,  0,  4, -1},
            { 7, -1,  6,  0, -1},
            {-1,  3,  9,  2,  0}
    };

    // 记录到每个顶点的路径距离（非最短）
    private int[] dist;
    // 记录每个顶点的前驱顶点（用于追溯最短路径）
    private int[] path;

    public void dijkstra(int node) {
        dist = new int[graph.length];
        path = new int[graph.length];
        // 记录当前顶点是否已经是最短路径
        boolean[] flag = new boolean[graph.length];
        // 初始化
        for (int i = 0; i < graph.length; i++) {
            dist[i] = graph[node][i];
            path[i] = 0;
            flag[i] = false;
        }
        // 初始顶点本身初始化
        dist[node] = 0;
        flag[node] = true;
        int i = node;
        // 寻找最短路径
        for (int count = 1; count < graph.length; count++) {
            int min = Integer.MAX_VALUE;
            int minIndex = -1;
            for (int j = 0; j < graph.length; j++) {
                int weight = graph[i][j];
                // i j 间存在路径且源点到 j 的权重大于源点到 i 的权重加 i j 间的权重
                // 并且源点到 j 不是最短路径
                if (weight > -1 && (weight + dist[i] < dist[j] || dist[j] < 0)
                        && !flag[j]) {
                    // 更新源点到 j 的路径权重
                    dist[j] = weight + dist[i];
                    // 将 j 的前驱节点记为 i
                    path[j] = i;
                }
                // 记录本轮遍历过程中的路径权重最小的节点
                if (dist[j] > 0 && dist[j] < min && !flag[j]) {
                    min = dist[j];
                    minIndex = j;
                }
            }
            // 本轮权重最小节点的访问标识置为 true
            flag[minIndex] = true;
            // 最小节点作为下一轮的前驱节点
            i = minIndex;
        }
    }

    public void print() {
        for (int i = 0; i < path.length; i++) {
            int cur = i;
            StringBuilder sb = new StringBuilder().append((char) (cur + 'A'));
            while (path[cur] != cur) {
                cur = path[cur];
                sb.insert(0, (char) (cur + 'A') + " -> ");
            }
            System.out.println(dist[i] + ": " + sb);
        }
    }
    public static void main(String[] args) {
        Dijkstra dijkstra = new Dijkstra();
        dijkstra.dijkstra(0);
        dijkstra.print();
    }
}

