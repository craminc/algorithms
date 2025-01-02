import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

import static math.Calc.lcm;

public class Draft {

    static class NumArray {

        private int[] bit;

        public NumArray(int[] nums) {
            bit = new int[nums.length + 1];
            for (int i = 0; i < nums.length; i++) {
                add(i, nums[i]);
            }
        }

        public void add(int index, int val) {
            for (int i = index + 1; i < bit.length; i += lowbit(i)) {
                bit[i] += val;
            }
        }

        public void update(int index, int val) {
            add(index, val - sumRange(index, index));
        }

        public int sumRange(int left, int right) {
            return sum(right + 1) - sum(left);
        }

        public int sum(int end) {
            int sum = 0;
            for (int i = end; i > 0; i -= lowbit(i)) {
                sum += bit[i];
            }
            return sum;
        }

        public int lowbit(int x) {
            return x & -x;
        }
    }

    public static void main(String[] args) {
        Draft draft = new Draft();
        long start = System.currentTimeMillis();
        draft.longestNiceSubarray(new int[]{
                178830999,19325904,844110858,806734874,280746028,
                64,256,33554432,882197187,104359873,453049214,
                820924081,624788281,710612132,839991691
        });
        long end = System.currentTimeMillis();
        System.out.println(end - start);
    }

    public int longestNiceSubarray(int[] nums) {
        int max = 1;
        int l = 0, r = 0;
        int bit = 0;
        while (r < nums.length) {
            if ((bit & nums[r]) == 0) {
                bit |= nums[r];
                max = Math.max(max, r - l + 1);
            } else {
                int i = r - 1;
                bit = nums[r];
                while (i > l && (nums[i] & nums[r]) == 0)
                    bit |= nums[i--];
                l = i + 1;
            }
            r++;
        }

        return max;
    }

    public long makeIntegerBeautiful(long n, int target) {
        int[] arr = new int[13];
        int sum = 0, idx = -1;
        while (n > 0) {
            arr[++idx] = (int) (n % 10);
            sum += arr[idx];
            n /= 10;
        }

        int k = -1, add = 0;
        long m = 1;
        while (arr[++k] == 0) m *= 10;
        arr[k]--;
        while (sum > target) {
            add += (10 - arr[k] - 1) * m;
            sum -= arr[k];
            k++;
            m *= 10;
        }

        return add;
    }

    public int maxStarSum(int[] vals, int[][] edges, int k) {
        int n = vals.length;
        int[][] g = new int[n][n];
        for (int[] edge : edges) {
            int x = edge[0], y = edge[1];
            g[x][y] = vals[y]; g[y][x] = vals[x];
        }
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int sum = vals[i];
            Arrays.sort(g[i]);
            int j = n - 1;
            int t = k;
            while (t-- > 0 && g[i][j] > 0) {
                sum += g[i][j--];
            }
            max = Math.max(max, sum);
        }

        return max;
    }

    public int beautySum(String s) {
        char[] c = s.toCharArray();
        int n = c.length;
        int[][] pre = new int[n + 1][26];
        int res = 0;
        for (int i = 1; i <= n; i++) {
            int idx = c[i - 1] - 'a';
            for (int p = i; p <= n; p++) pre[p][idx]++;
            for (int j = 0; j < i; j++) {
                int max = 0;
                int min = n;
                for (int k = 0; k < 26; k++) {
                    int sum = pre[i][k] - pre[j][k];
                    if (sum == 0) continue;
                    max = Math.max(max, sum);
                    min = Math.min(min, sum);
                }
                res += max - min;
            }
        }

        return res;
    }

    public int minOperations(int[] nums1, int[] nums2) {
        // count sort
        int[] cnt1 = new int[7], cnt2 = new int[7];
        for (int num1 : nums1)  cnt1[num1]++;
        for (int num2 : nums2)  cnt2[num2]++;
        // sum
        int diff = 0;
        for (int i = 1; i < 7; i++)
            diff += i * (cnt1[i] - cnt2[i]);
        if (diff < 0) {
            int[] tmp = cnt1;
            cnt1 = cnt2;
            cnt2 = tmp;
        }
        if (diff == 0) return 0;
        int l = 1, r = 6;
        int res = 0;
        while (diff > 0 && r > l) {
            while (r > l && cnt1[r] == 0) r--;
            while (r > l && cnt2[l] == 0) l++;
            diff -= r - l;
            cnt1[r]--; cnt2[l]--;
            res++;
        }
        return res == 0 ? -1 : res;
    }

    public static class FreqStack {

        private List<Deque<Integer>> stacks = new ArrayList<>();
        private Map<Integer, Integer> cntMap = new HashMap<>();

        public FreqStack() {}

        public void push(int val) {
            int cnt = cntMap.getOrDefault(val, 0) + 1;
            cntMap.put(val, cnt);
            if (stacks.size() < cnt) {
                stacks.add(new ArrayDeque<>());
            }
            Deque<Integer> stack = stacks.get(cnt - 1);
            stack.push(val);
        }

        public int pop() {
            int idx = stacks.size() - 1;
            Deque<Integer> stack = stacks.get(idx);
            int res = stack.pop();
            if (stack.isEmpty()) stacks.remove(idx);
            cntMap.put(res, cntMap.get(res) - 1);
            return res;
        }
    }

    public double largestSumOfAverages(int[] nums, int k) {
        int n = nums.length;
        // prefix sum
        double[] pre = new double[n + 1];
        for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + nums[i];
        // dp[i][j] means split 0 - i to j part
        double[][] dp = new double[n + 1][k + 1];
        for (int j = 1; j <= k; j++) {
            for (int i = j; i <= n; i++) {
                for (int l = j - 1; l < i; l++) {
                    dp[i][j] = Math.max(dp[i][j],
                            dp[l][j - 1] + (pre[i] - pre[l]) / (i - l));
                    if (l == 0) break;
                }
            }
        }
        return dp[n][k];
    }

    public int reachableNodes(int[][] edges, int maxMoves, int n) {
        // build graph by edges
        int[][] graph = new int[n][n];
        for (int[] edge : edges) {
            int x = edge[0], y = edge[1];
            graph[x][y] = graph[y][x] = edge[2] + 1;
        }
        // record min distance
        int[] dist = new int[n];
        // note whether been visited
        boolean[] vis = new boolean[n];
        // dijkstra
        for (int i = 0, cnt = 0; cnt < n; cnt++) {
            int m = -1;
            // use node i as transit node
            for (int j = 1; j < n; j++) {
                // target node must not been visited
                if (vis[j]) continue;
                int w = graph[i][j];
                // has shorter path
                if (w > 0 && (dist[j] == 0 || dist[j] > dist[i] + w))
                    //update distance
                    dist[j] = dist[i] + w;
                if (dist[j] > 0 && (m == -1 || dist[j] < dist[m]))
                    m = j;
            }
            // min distance is more than maxMoves
            if (m == -1 || dist[m] >= maxMoves) break;
            // chose closest node as next
            i = m;
            vis[i] = true;
        }

        int res = 0;
        for (int i = 0; i < n; i++)
            if (dist[i] > 0 && dist[i] <= maxMoves) res++;
        for (int[] edge : edges) {
            int d1 = dist[edge[0]], d2 = dist[edge[1]], w = edge[2];
            // record how many split node can reach
            int c1 = 0, c2 = 0;
            if (d1 > 0 && d1 < maxMoves)
                c1 = Math.min(w, maxMoves - d1);
            if (d2 > 0 && d2 < maxMoves)
                c2 = Math.min(w, maxMoves - d2);
            res += Math.min(w, c1 + c2);
        }

        return res;
    }

    public int countBalls(int lowLimit, int highLimit) {
        int[] map = new int[40];
        for (int i = lowLimit; i <= highLimit; i++) {
            int idx = 0;
            while (i > 0) {
                idx += i % 10;
                i /= 10;
            }
            map[idx]++;
        }
        int max = 0;
        for (int m : map) max = Math.max(max, m);
        return max;
    }

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Map<Integer, Integer> map = discretize(nums);
        // use BIT to record cnt the cnt of elem less than nums[i]
        // and update while traversing
        bit = new int[map.size() + 1];
        for (int i = n - 1; i >= 0; i--) {
            int idx = map.get(nums[i]) + 1;
            res[i] = query(idx);
            // update cnt
            update(idx, 1);
        }
        List<Integer> ls = new ArrayList<>();
        for (int r : res) ls.add(r);
        return ls;
    }

    private Map<Integer, Integer> discretize(int[] nums) {
        int n = nums.length;
        int[] tmp = Arrays.copyOf(nums, n);
        Arrays.sort(tmp);
        int idx = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(tmp[0], idx++);
        for (int i = 1; i < n; i++) {
            while (i < n && tmp[i] == tmp[i - 1]) i++;
            if (i == n) break;
            map.put(tmp[i], idx++);
        }
        return map;
    }

    private int[] bit;

    private void update(int idx, int val) {
        while (idx < bit.length) {
            bit[idx] += val;
            idx += lowbit(idx);
        }
    }

    private int query(int idx) {
        int sum = 0;
        while (idx > 0) {
            sum += bit[idx];
            idx -= lowbit(idx);
        }
        return sum;
    }

    private int lowbit(int x) {
        return x & -x;
    }

    public int subarrayLCM(int[] nums, int k) {
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n; i++) {
            int m = nums[i];
            for (int j = i; j < n; j++) {
                m = lcm(m, nums[j]);
                if (m == k) res++;
            }
        }
        return res;
    }

    public int splitArray(int[] nums, int k) {
        int down = 0, top = 0;
        for (int num : nums) {
            top += num;
            down = Math.max(num, down);
        }

        while (down < top) {
            int mid = down + top >> 1;
            if (check(nums, k, mid)) top = mid;
            else down = mid + 1;
        }

        return down;
    }

    public boolean check(int[] nums, int m, int max) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
            if (sum > max) {
                sum = num;
                m--;
            }
        }
        return m > 0;
    }

    public boolean robot(String command, int[][] obstacles, int x, int y) {
        char[] cmds = command.toCharArray();
        int u = 0, r = 0;
        for (int cmd : cmds) {
            if (cmd == 'U') u++;
            else if (cmd == 'R') r++;
        }
        // record the position where robot will reach
        int[][] map = new int[r + 1][u + 1];
        u = 0; r = 0;
        for (int cmd : cmds) {
            if (cmd == 'U') map[r][++u] = 1;
            else if (cmd == 'R') map[++r][u] = 1;
        }
        // judge if end can reach
        if (!canReach(map, x, y)) return false;
        // judge if obstacles can reach
        for (int[] ob : obstacles) {
            if (ob[0] > x || ob[1] > y) continue;
            if (canReach(map, ob[0], ob[1])) return false;
        }
        return true;
    }

    public boolean canReach(int[][] map, int x, int y) {
        int r = map.length - 1;
        int u = map[0].length - 1;

        int rd = (x + y) / (r + u);
        int rx = x - rd * r;
        int ry = y - rd * u;
        return rx <= r && ry <= u && map[rx][ry] == 1;
    }

    public int countSubstrings(String s) {
        char[] c = s.toCharArray();
        int n = c.length;
        // dp[i][j] record if c[i-j] is palindrome
        boolean[][] dp = new boolean[n][n];
        int cnt = 0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                dp[i][j] = dp[i] == dp[j] && (i + 1 > j - 1 || dp[i + 1][j - 1]);
                cnt += dp[i][j] ? 1 : 0;
            }
        }
        return cnt;
    }

    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            while(nums[i] != i + 1) {
                int idx = nums[i] - 1;
                if (nums[i] == nums[idx]) break;
                int tmp = nums[idx];
                nums[idx] = nums[i];
                nums[i] = tmp;
            }
        }
        for (int i = 0; i < nums.length; i++)
            if (nums[i] != i + 1) res.add(i + 1);
        return res;
    }

    public int[] bestCoordinate(int[][] towers, int radius) {
        int[] res = new int[2];

        int maxX = 0, maxY = 0;
        // find maxX and maxY
        for (int[] tower : towers) {
            maxX = Math.max(maxX, tower[0]);
            maxY = Math.max(maxY, tower[1]);
        }
        // traverse all position in the area bounded by maxX-maxY
        int maxSignal = 0;
        for (int i = 0; i <= maxX; i++) {
            for (int j = 0; j <= maxY; j++) {
                int signal = 0;
                for (int[] tower : towers) {
                    int x = tower[0] - i;
                    int y = tower[1] - j;
                    int dist = (int) Math.sqrt(x * x + y * y);
                    // calc the signal
                    signal += dist > radius ? 0 : (tower[2] / (1 + dist));
                }
                if (maxSignal < signal) {
                    maxSignal = signal;
                    res[0] = i;
                    res[1] = j;
                }
            }
        }

        return res;
    }

    public int maximumDetonation(int[][] bombs) {
        int[] vis = new int[bombs.length];
        // virtual bomb connected all bombs
        int[] bomb = new int[]{0, 0, -1};
        // dfs can find all relate bombs
        return findBomb(bombs, bomb, vis) - 1;
    }

    public int findBomb(int[][] bombs, int[] bomb, int[] vis) {
        int max = 0;
        for (int i = 0; i < bombs.length; i++) {
            if (vis[i] == 0 && isCon(bomb, bombs[i])) {
                vis[i] = 1;
                max = Math.max(max, findBomb(bombs, bombs[i], vis));
                vis[i] = 0;
            }
        }
        return max + 1;
    }

    public boolean isCon(int[] bomb1, int[] bomb2) {
        if (bomb1[2] == -1) return true;
        long x = bomb1[0] - bomb2[0];
        long y = bomb1[1] - bomb2[1];
        long l = bomb1[2];
        return x * x + y * y <= l * l;
    }

    private static int[] readInput() {
        File file = new File("input.txt");
        byte[] input = new byte[209227];
        try (InputStream is = Files.newInputStream(file.toPath())) {
            int cnt = is.read(input);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        String s = new String(input);
        String[] split = s.split(",");
        int[] res = new int[split.length];
        int i = 0;
        for (String sl : split) {
            res[i++] = Integer.parseInt(sl);
        }
        return res;
    }
    public int shortestSubarray(int[] nums, int k) {
        int n = nums.length;
        int[] pre = new int[n + 1];
        for (int i = 0; i < n; i++) {
            pre[i + 1] = pre[i] + nums[i];
        }
        int minL = 100001;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (pre[j] == -1) continue;
                if (pre[i] - pre[j] >= k) {
                    minL = Math.min(minL, i - j);
                    pre[j] = -1;
                }
                if (pre[j] >= pre[i]) pre[j] = -1;
            }
        }
        return minL > n ? -1 : minL;
    }
    public int shortestSubarray1(int[] nums, int k) {
        int n = nums.length;
        long[] preSumArr = new long[n + 1];
        for (int i = 0; i < n; i++) {
            preSumArr[i + 1] = preSumArr[i] + nums[i];
        }
        int res = n + 1;
        Deque<Integer> queue = new ArrayDeque<Integer>();
        for (int i = 0; i <= n; i++) {
            long curSum = preSumArr[i];
            while (!queue.isEmpty() && curSum - preSumArr[queue.peekFirst()] >= k) {
                res = Math.min(res, i - queue.pollFirst());
            }
            while (!queue.isEmpty() && preSumArr[queue.peekLast()] >= curSum) {
                queue.pollLast();
            }
            queue.offerLast(i);
        }
        return res < n + 1 ? res : -1;
    }

    private int[][] dirs = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    public int shortestBridge(int[][] grid) {
        // dfs + bfs
        int n = grid.length;
        int sx = 0, sy = 0;
        f1:for (int i = 0; i < n; i++) {
            f2:for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    sx = i; sy = j;
                    break f1;
                }
                grid[i][j] = -1;
            }
        }
        Deque<int[]> queue = new ArrayDeque<>();
        // first use dfs to find all land of island1
        dfs(grid, sx, sy, queue);
        // then use bfs to expand island1 to attach island2
        int level = 0;
        while (!queue.isEmpty()) {
            int sz = queue.size();
            for (int i = 0; i < sz; i++) {
                int[] pos = queue.poll();
                for (int[] dir : dirs) {
                    int nx = pos[0] + dir[0];
                    int ny = pos[1] + dir[1];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= n ||
                            grid[nx][ny] == -1) continue;
                    if (grid[nx][ny] == 0) {
                        grid[nx][ny] = 1;
                        queue.offer(new int[]{nx, ny});
                    }
                    // once reach, the res is the bfs level
                    else return level;
                }
            }
            level++;
        }
        return 0;
    }

    private void dfs(int[][] grid, int x, int y, Deque<int[]> queue) {
        int n = grid.length;
        if (x < 0 || x >= n || y < 0 || y >= n || grid[x][y] != 1) return;
        grid[x][y] = -1;
        queue.offer(new int[]{x, y});
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            dfs(grid, nx, ny, queue);
        }
    }


    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        // dp[i] represent the max profit can get from the first I job
        int[] dp = new int[n + 1];
        // the index of job
        Integer[] ji = new Integer[n];
        for (int i = 0; i < n; i++) ji[i] = i;
        // sort by the endTime
        Arrays.sort(ji, (a, b) -> endTime[a] - endTime[b]);

        for (int i = 0; i < n; i++) {
            // use dichotomy to find the nearest job j where endTime[j] <= startTime[i]
            int l = 0, r = i - 1;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (endTime[ji[mid]] <= startTime[ji[i]]) l = mid;
                else r = mid - 1;
            }
            // chose job[i] or not
            dp[i + 1] = Math.max(dp[i], dp[l + 1] + profit[ji[i]]);
        }

        return dp[n];
    }

    private List<List<String>> res1 = new ArrayList<>();
    public List<List<String>> findLadders(String b, String e, List<String> ws) {
        set.addAll(ws);
        if (!set.contains(e)) return res1;
        twoWayBfs(b, e);
        return res1;
    }

    // two-way bfs
    public void twoWayBfs(String b, String e) {
        // two queue: q1 order, q2 reverse order
        Deque<String> q1 = new ArrayDeque<>();
        Deque<String> q2 = new ArrayDeque<>();
        // add begin and end
        q1.offer(b);
        q2.offer(e);
        // two map record the length of the path
        Map<String, List<List<String>>> m1 = new HashMap<>();
        Map<String, List<List<String>>> m2 = new HashMap<>();
        List<String> l1 = new ArrayList<>(), l2 = new ArrayList<>();
        l1.add(b); l2.add(e);
        List<List<String>> ls1 = new ArrayList<>(), ls2 = new ArrayList<>();
        ls1.add(l1); ls2.add(l2);
        m1.put(b, ls1);
        m2.put(e, ls2);

        while (!q1.isEmpty() && !q2.isEmpty()) {
            boolean find = false;
            // make the balance of two bfs
            if (q1.size() <= q2.size()) {
                find = update(q1, m1, m2, true);
            } else {
                find = update(q2, m2, m1, false);
            }
            if (find) return;
        }
    }

    boolean update(Deque<String> queue, Map<String, List<List<String>>> cur,
                   Map<String, List<List<String>>> other, boolean order) {
        int m = queue.size();
        // traverse all candidate
        boolean find = false;
        while (m-- > 0) {
            String str = queue.poll();
            char[] word = str.toCharArray();
            for (int i = 0; i < word.length; i++) {
                for (int j = 0; j < 26; j++) {
                    char wi = word[i];
                    word[i] = (char) (j + 'a');
                    if (word[i] == wi) continue;
                    String newStr = new String(word);
                    word[i] = wi;
                    if (set.contains(newStr)) {
                        if (cur.containsKey(newStr) &&
                                cur.get(newStr).get(0).size() != cur.get(str).get(0).size() + 1)
                            // has visited and the pre path is not longer
                            continue;
                        if (other.containsKey(newStr)) {
                            // find the path
                            List<List<String>> ls1 = cur.get(str);
                            List<List<String>> ls2 = other.get(newStr);
                            for (List<String> l1 : ls1) {
                                for (List<String> l2 : ls2) {
                                    List<String> path = new ArrayList<>();
                                    if (order) {
                                        path.addAll(l1);
                                        path.addAll(l2);
                                    } else {
                                        path.addAll(l2);
                                        path.addAll(l1);
                                    }
                                    res1.add(path);
                                }
                            }
                            find = true;
                        } else {
                            // add node to the path
                            queue.offer(newStr);
                            List<List<String>> ls = cur.get(str);
                            List<List<String>> nls = new ArrayList<>();
                            for (List<String> l : ls) {
                                List<String> n = new ArrayList<>(l);
                                n.add(newStr);
                                nls.add(n);
                            }
                            List<List<String>> ols = cur.get(newStr);
                            if (ols == null)  cur.put(newStr, nls);
                            else ols.addAll(nls);
                        }
                    }
                }
            }
        }
        // not find integrated path
        return find;
    }

    public boolean isPalindrome(String s) {
        char[] c = s.toCharArray();
        int l = 0, r = c.length - 1;
        while (l < r) {
            while (l < r && !isWord(c[l])) l++;
            while (l < r && !isWord(c[r])) r--;
            if (c[l] != c[r] && Math.abs(c[l] - c[r]) != 32)
                return false;
            l++; r--;
        }
        return true;
    }

    private boolean isWord(char c) {
        return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z';
    }

    public int maximalRectangle(char[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        int[] heights = new int[col];
        int res = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == 0) heights[j] = 0;
                else heights[j] += matrix[i][j];
            }
            res = Math.max(res, largestRectangleArea(heights));
        }
        return res;
    }

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] nh = new int[n + 2];
        for (int i = 0; i < n; i++) nh[i + 1] = heights[i];
        int[] stack = new int[n + 2];
        int sp = -1;

        int max = 0;
        for (int i = 0; i < n + 2; i++) {
            while (sp >= 0 && nh[stack[sp]] > nh[i]) {
                max = Math.max(max, nh[stack[sp--]] * (i - stack[sp] - 1));
            }
            stack[++sp] = i;
        }
        return max;
    }

    public ListNode partition(ListNode head, int x) {
        head = new ListNode(0, head);
        ListNode cur = head;
        ListNode p = cur;
        while (cur != null && cur.next != null) {
            if (cur.next.val < x) {
                if (p != cur) {
                    ListNode pn = p.next;
                    p.next = cur.next;
                    cur.next = cur.next.next;
                    p.next.next = pn;
                    p = p.next;
                    continue;
                }
                p = p.next;
            }
            cur = cur.next;
        }
        return head.next;
    }

    private ListNode buildList(int[] nums) {
        ListNode head = new ListNode(0);
        ListNode cur = head;
        for (int num : nums) {
            cur.next = new ListNode(num);
            cur = cur.next;
        }
        return head.next;
    }

    public int scoreOfParentheses(String s) {
        int[] stack = new int[s.length()];
        int sp = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') sp++;
            else stack[sp - 1] += Math.max(2 * stack[sp--], 1);
        }
        return stack[sp];
    }

    public void sortColors(int[] nums) {
        int p0 = 0, p1 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                nums[i] = nums[p0];
                nums[p0++] = 0;
            }
            if (nums[i] == 1) {
                nums[i] = nums[p1];
                nums[p1++] = 1;
            }
        }
    }

    public int mySqrt(int x) {
        if (x == 1) return 1;
        int l = 0;
        int r = Math.min(x / 2, 65535);
        while (l < r) {
            int mid = l + r + 1 >> 1;
            int val = mid * mid;
            if (val <= x) l = mid;
            else r = mid - 1;
        }
        return l;
    }

    public String addBinary(String a, String b) {
        char[] ar = a.toCharArray();
        char[] br = b.toCharArray();
        int al = ar.length;
        int bl = br.length;
        int len = Math.max(al, bl);
        char[] arr = new char[len + 1];

        int cb = 0;
        for (int i = 0; i <= len; i++) {
            int ai = i < al ? ar[al - i - 1] - '0' : 0;
            int bi = i < bl ? br[bl - i - 1] - '0' : 0;
            int n = ai + bi + cb;
            cb = n / 2;
            arr[i] = (char) (n % 2 + '0');
        }
        return new String(arr);
    }

    public String getPermutation(int n, int k) {
        int[] vis = new int[n];
        int cnt = 1;
        int res = 0;
        k--;
        for (int i = n - 1; i > 0; i--) cnt *= i;
        for (int i = n - 1; i >= 0; i--) {
            int idx = k / cnt;
            if (i > 0) {
                k %= cnt;
                cnt /= i;
            }
            int t = -1;
            while (idx >= 0) {
                if (vis[++t] == 0) idx--;
            }
            vis[t] = 1;
            res = res * 10 + t + 1;
        }
        return String.valueOf(res);
    }

    public double myPow(double x, int n) {
        int c = Math.abs(n);
        double ret = 1f;
        double cx = x;
        while (c > 0) {
            if ((n & 1) == 1) ret *= cx;
            cx *= cx;
            c = c >> 1;
        }
        return n > 0 ? ret : 1f / ret;
    }

    public String multiply(String num1, String num2) {
        char[] c1 = num1.toCharArray();
        char[] c2 = num2.toCharArray();
        int[] res = new int[c1.length + c2.length];

        for (int i = c1.length - 1; i >= 0; i--) {
            int n1 = c1[i] - '0';
            for (int j = c2.length - 1; j >= 0; j--) {
                int n2 = c2[j] - '0';
                int m = n1 * n2 + res[i + j + 1];
                res[i + j + 1] = m % 10;
                res[i + j] += m / 10;
            }
        }

        char[] ans;
        int j = 0;
        if (res[0] == 0) {
            ans = new char[res.length - 1];
            j++;
        }
        else ans = new char[res.length];

        for (int i = 0; i < ans.length; i++) {
            ans[i] = (char) ('0' + res[j++]);
        }
        return new String(ans);
    }

    public int getKthMagicNumber(int k) {
        // record use how many
        int p3 = 0;
        int p5 = 0;
        int p7 = 0;
        int[] res = new int[k];
        res[0] = 1;
        for (int i = 1; i < k; i++) {
            int n3 = 3 * res[p3];
            int n5 = 5 * res[p5];
            int n7 = 7 * res[p7];
            int n = Math.min(n3, Math.min(n5, n7));
            res[i] = n;
            if (n3 == n) p3++;
            if (n5 == n) p5++;
            if (n7 == n) p7++;
        }
        return res[k - 1];
    }

    public int longestValidParentheses(String s) {
        int n = s.length();
        char[] arr = s.toCharArray();
        int max = 0;
        // stack record the idx
        int[] stack = new int[n + 1];
        int sp = 0;
        stack[sp++] = -1;

        for (int i = 0; i < n; i++) {
            if (arr[i] == '(') stack[sp++] = i;
            else {
                sp--;
                if (sp == 0) stack[sp++] = i;
                else max = Math.max(max, i - stack[sp - 1]);
            }
        }
        return max;
    }

    public int[] missingTwo(int[] nums) {
        int n = nums.length + 2;
        // 1 + 2 + ... + n
        int sum = (1 + n) * n >> 1;
        // 1*1 + 2*2 + ... + n*n
        long powSum = n * (n + 1) * (2L * n + 1) / 6;
        for (int num : nums) {
            sum -= num;
            powSum -= (long) num * num;
        }
        // a + b
        int apb = sum;
        // a*a + b*b
        int pApb = (int) powSum;
        // a * b
        int amb = (apb * apb - pApb) >> 1;
        // a - b
        int adb = (int) Math.sqrt(pApb - 2 * amb);

        int a = apb + adb >> 1;
        int b = apb - a;

        return new int[]{a, b};
    }

    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length < 1) return null;
        // merge sort
        int n = lists.length;
        int i = 0;
        while (n > 1) {
            lists[i] = merge(lists, i);
            i++;
            if (i * 2 >= n) {
                n = (n + 1) / 2;
                i = 0;
            }
        }
        return lists[0];
    }

    public ListNode merge(ListNode[] lists, int idx) {
        if (idx * 2 + 1 == lists.length) return lists[idx * 2];
        ListNode p = lists[idx * 2];
        ListNode q = lists[idx * 2 + 1];
        ListNode head = new ListNode();
        ListNode cur = head;
        while (p != null && q != null) {
            if (p.val < q.val) {
                cur.next = p;
                p = p.next;
            } else {
                cur.next = q;
                q = q.next;
            }
            cur = cur.next;
        }
        if (q == null) cur.next = p;
        else cur.next = q;
        return head.next;
    }

    public int kSimilarity(String s1, String s2) {
        // bfs the steps for s1 state change to s2 state
        // each step swap a char of s1 where s1[i] != s2[i]
        // this queue record a temp state of s1
        char[] s2c = s2.toCharArray();
        int n = s1.length();
        Deque<String> queue = new ArrayDeque<>();
        // original state
        queue.offer(s1);
        // record the round of bfs
        int r = 0;
        // record the position where s1[i] != s2[i]

        while (!queue.isEmpty()) {
            // use this loop can split rounds
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                String s = queue.poll();
                char[] sc = s.toCharArray();
                if (s.equals(s2)) return r;
                int idx = r;
                while (idx < n && sc[idx] == s2c[idx]) idx++;
                int cur = idx;
                while (cur < n) {
                    if (sc[cur] == s2c[idx] && sc[cur] != s2c[cur]) {
                        swap(sc, cur, idx);
                        queue.offer(new String(sc));
                        swap(sc, cur, idx);
                    }
                    cur++;
                }
            }
            r++;
        }
        return -1;
    }

    private void swap(char[] arr, int i, int j) {
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    private List<List<String>> res = new ArrayList<>();
    private Set<String> set = new HashSet<>();
    public List<String> wordBreak(String s, List<String> wordDict) {
        for (String w : wordDict) set.add(w);
        backtrace(new ArrayList<>(), s, 0);
        return res.stream().map(t -> String.join(" ", t)).collect(Collectors.toList());
    }

    public void backtrace(List<String> path, String s, int idx) {
        if (idx == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        int end = idx + 1;
        while (end <= s.length()) {
            String sub = s.substring(idx, end);
            if (set.contains(sub)) {
                path.add(sub);
                backtrace(path, s, end);
                path.remove(path.size() - 1);
            }
            end++;
        }
    }
}
