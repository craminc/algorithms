package graph;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.TimeUnit;

/**
 * A Star(A*) is an optimisation of Uniform Cost Search(UCS), it uses a heuristic
 * function to reduce the time cost of pathfinder
 */
public class AStar extends UniformCostSearch {

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        AStar aStar = new AStar();
        aStar.search(new Location(4,1), new Location(5, 8));
        long end = System.currentTimeMillis();
        System.out.println(end - start);
    }

    @Override
    public void search(Location start, Location end) {
        // store candidate node
        PriorityQueue<Node> queue = new PriorityQueue<>(Comparator.comparingInt(t -> t.cost));
        queue.add(new Node(start, 0));
        // record the distance from start to current
        Map<Location, Integer> dist = new HashMap<>();
        dist.size();
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
            for (Location next : super.neighbors(prior.location)) {
                // calc the cost from start to next
                int newCost = dist.get(cur) + cost(next);
                // if the
                if (!dist.containsKey(next) || dist.get(next) > newCost) {
                    // here is the different
                    dist.put(next, newCost);
                    from.put(next, cur);
                    queue.add(new Node(next, newCost + heuristic(next, end)));
                }
            }
        }
        super.reconstruct(from, dist, end);
    }

    public int heuristic(Location next, Location end) {
        return Math.abs(end.x - next.x) + Math.abs(end.y - next.y);
    }
}
