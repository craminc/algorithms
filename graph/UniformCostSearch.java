package graph;

import java.util.*;

public class UniformCostSearch {

    public final static int[][] DIRS = new int[][]{{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
    public final static int[][] GRID = new int[][]{
            {1, 1 , 1 , 1 , 1, 1, 1, 1, 1, 1},
            {1, 1 , 1 , 1 , 5, 5, 1, 1, 1, 1},
            {1, 1 , 1 , 1 , 5, 5, 5, 1, 1, 1},
            {1, 1 , 1 , 1 , 5, 5, 5, 5, 1, 1},
            {1, 1 , 1 , 5 , 5, 5, 5, 5, 1, 1},
            {1, 1 , 1 , 5 , 5, 5, 5, 5, 1, 1},
            {1, 1 , 1 , 1 , 5, 5, 5, 1, 1, 1},
            {1, -1, -1, -1, 5, 5, 5, 1, 1, 1},
            {1, -1, -1, -1, 5, 5, 1, 1, 1, 1},
            {1, 1 , 1 , 1 , 1, 1, 1, 1, 1, 1}
    };

    public final static int LMT = 10;

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        UniformCostSearch ucs = new UniformCostSearch();
        ucs.search(new Location(4,1), new Location(5, 8));
        long end = System.currentTimeMillis();
        System.out.println(end - start);
    }

    public void search(Location start, Location end) {
        // store candidate node
        PriorityQueue<Node> queue = new PriorityQueue<>(Comparator.comparingInt(t -> t.cost));
        queue.add(new Node(start, 0));
        // record the distance from start to current
        Map<Location, Integer> dist = new HashMap<>();
        // record the previous location of current(use to reconstruct route)
        Map<Location, Location> from = new HashMap<>();
        dist.put(start, 0);
        from.put(start, start);
        while (!queue.isEmpty()) {
            // poll the prior node
            Node prior = queue.poll();
            Location cur = prior.location;
            if (cur.equals(end)) break;
            // find neighbors
            for (Location next : this.neighbors(prior.location)) {
                // calc the cost from start to next
                int newCost = dist.get(cur) + cost(next);
                // if the
                if (!dist.containsKey(next) || dist.get(next) > newCost) {
                    dist.put(next, newCost);
                    from.put(next, cur);
                    queue.add(new Node(next, newCost));
                }
            }
        }
        this.reconstruct(from, dist, end);
    }

    protected List<Location> neighbors(Location cur) {
        List<Location> ls = new ArrayList<>();
        for (int[] dir : DIRS) {
            int nx = cur.x + dir[0];
            int ny = cur.y + dir[1];
            if (nx >= 0 && nx < LMT && ny >= 0 && ny < LMT && GRID[nx][ny] > 0)
                ls.add(new Location(nx, ny));
        }
        return ls;
    }

    protected int cost(Location next) {
        return GRID[next.x][next.y];
    }

    protected void reconstruct(Map<Location, Location> from, Map<Location, Integer> dist, Location end) {
        System.out.println(dist.get(end));
        StringBuilder sb = new StringBuilder();
        while (from.get(end) != end && from.get(end) != null) {
            sb.insert(0, "->" + end);
            end = from.get(end);
        }
        sb.insert(0, end);
        System.out.println(sb);
    }

    public static class Location {
        int x;
        int y;

        public Location(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Location location = (Location) o;
            return x == location.x && y == location.y;
        }

        @Override
        public int hashCode() {
            int result = x;
            result = 31 * result + y;
            return result;
        }

        @Override
        public String toString() {
            return String.format("(%d, %d)", x, y);
        }
    }

    public static class Node {
        Location location;
        int cost;

        public Node(Location location, int cost) {
            this.location = location;
            this.cost = cost;
        }
    }
}
