import tree.TreeNode;
import util.BuildArray;
import util.BuildTree;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class Draft3 {

    public final static int[] prime = new int[169];
    static {
        int idx = 1;
        lp: for (int i = 2; i <= 1000; i++) {
            for (int j = 2; j * j <= i; j++)
                if (i % j == 0) continue lp;
            prime[idx++] = i;
        }
    }

    public boolean primeSubOperation(int[] nums) {
        int pre = 0;
        for (int i = 0; i < nums.length; i++) {
            int dif = nums[i] - pre;
            // 从prime中找到小于等于dif的第一个元素
            int l = 0, r = 167;
            while (l < r) {
                int m = l + r + 1 >> 1;
                if (prime[m] >= dif) r = m - 1;
                else l = m;
            }
            nums[i] -= prime[l];
            if (nums[i] <= pre) return false;
            pre = nums[i];
        }

        return true;
    }

    public List<Long> minOperations(int[] nums, int[] queries) {
        List<Long> res = new ArrayList<>();
        int n = nums.length;
        Arrays.sort(nums);
        long[] pre = new long[n + 1];
        // 计算前缀和
        for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + nums[i];
        // 二分查找第一个大于query的num的idx
        for (int query : queries) {
            int l = 0, r = n - 1;
            while (l < r) {
                int m = l + r >> 1;
                if (nums[m] > query) r = m;
                else l = m + 1;
            }
            int cnt = nums[l] > query ? l : l + 1;
            long opt = pre[n] - 2 * pre[cnt] + (((long) cnt << 1) - n) * query;
            res.add(opt);
        }

        return res;
    }

//    public static int[][] dirs = new int[][]{
//            new int[]{1, 2}, new int[]{1, -2}, new int[]{-1, 2}, new int[]{-1, -2},
//            new int[]{2, 1}, new int[]{-2, 1}, new int[]{2, -1}, new int[]{-2, -1}
//    };
    public boolean checkValidGrid(int[][] grid) {
        int n = grid.length;
        int x = 0, y = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    x = i; y = j;
                    break;
                }
            }
        }

        for (int i = 0; i < n * n - 1; i++) {
            boolean v = false;
            for (int[] dir : dirs) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    if (grid[nx][ny] == i + 1) {
                        x = nx; y = ny;
                        v = true;
                        break;
                    }
                }
            }
            if (!v) return false;
        }
        return true;
    }

    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        int[] pre = new int[]{0, 0};
        int cnt = 0;
        for (int[] it : intervals) {
            if (it[0] >= pre[1]) pre = it;
            else {
                cnt++;
                if (it[1] < pre[1]) pre = it;
            }
        }

        return cnt;
    }

    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, Comparator.comparingInt(a -> a[0]));
        int cnt = 1;
        int end = points[0][1];
        for (int i = 1; i < points.length; i++) {
            int[] point = points[i];
            if (point[0] > end) cnt++;
            end = Math.min(end, point[1]);
        }

        return cnt;
    }

    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        Trie trie = new Trie();
        trie.addAll(products);
        List<List<String>> res = new ArrayList<>();
        Node cur = trie.root;
        String prefix = "";
        for (char s : searchWord.toCharArray()) {
            prefix += s;
            cur = cur.next[s - 'a'];
            res.add(trie.find(new ArrayList<>(), cur, prefix));
        }

        return res;
    }

    static class Trie {
        Node root = new Node();

        public void addAll(String[] words) {
            for (String word : words)
                add(word);
        }

        public void add(String word) {
            Node cur = root;
            for (char w : word.toCharArray()) {
                int idx = w - 'a';
                if (cur.next[idx] == null)
                    cur.next[idx] = new Node();
                cur = cur.next[idx];
            }
            cur.end = true;
        }

        public List<String> find(List<String> res, Node cur, String pre) {
            if (cur != null && res.size() < 3) {
                if (cur.end) res.add(pre);
                int idx = 0;
                for (Node nx : cur.next)
                    if (nx != null)
                        find(res, nx, pre + (char)('a' + idx));
            }
            return res;
        }
    }

    static class Node {
        boolean end;
        Node[] next = new Node[26];
    }

    public int longestDecomposition(String text) {
        char[] ch = text.toCharArray();
        int n = ch.length, k = 0;
        int l = 0, r = 0;

        while (l < (n >> 1)) {
            if (!same(ch, l, r)) r++;
            else {
                k++;
                l = ++r;
            }
        }

        return k;
    }

    public boolean same(char[] ch, int l, int r) {
        int len = r - l, n = ch.length - 1;
        for (int i = l, j = n - l - len; i <= r; i++, j++) {
            if (ch[i] != ch[j]) return false;
        }
        return true;
    }

    public String reverseWords(String s) {
        int n = s.length();
        char[] ch = s.toCharArray(), rs = new char[n + 1];

        int idx = n;
        for (int i = 0; i < n; i++) {
            if (ch[i] == ' ') continue;
            int j = i;
            while (j < n && ch[j] != ' ') j++;
            idx -= j - i;
            rs[idx] = ' ';
            int k = idx;
            while (i < j) rs[++k] = ch[i++];
            idx--;
        }

        return new String(rs);
    }

    public List<Boolean> camelMatch(String[] queries, String pattern) {
        // pattern 是 query 的删除部分小写字母的子序列
        List<Boolean> ls = new ArrayList<>();
        for (String query : queries)
            ls.add(this.match(pattern.toCharArray(), query.toCharArray()));
        return ls;
    }

    public boolean match(char[] pch, char[] qch) {
        int i = 0, n = pch.length;
        for (char c : qch) {
            if (i < n && c == pch[i]) i++;
            else if (c >= 'A' && c <= 'Z') return false;
        }
        return i == n;
    }

    public int longestOnes(int[] nums, int k) {
        int l = 0, r = 0, z = 0, n = nums.length;
        int max = 0;
        while (r < n) {
            while (r < n) {
                if (nums[r] == 0 && z <= k) z++;
                r++;
            }
            max = Math.max(max, r - l - 1);
            while (l < r && z > k) {
                if (nums[l++] == 0) z--;
            }
        }
        return max;
    }

    public int longestArithSeqLength(int[] nums) {
        int n = nums.length;
        // dp[i][j] 表示 nums[:i] 等差值为j的最长等差子序列
        int[][] dp = new int[n + 1][1001];

        int max = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                int dif = nums[i] - nums[j] + 500;
                dp[i + 1][dif] = dp[j][dif] + 1;
                max = Math.max(max, dp[i + 1][dif]);
            }
        }

        return max;
    }

    private boolean[] vis;
    private int res = 0;
    public int minReorder(int n, int[][] connections) {
        vis = new boolean[n];
        int[][] g = new int[n][n];
        for (int[] con : connections) {
            g[con[0]][con[1]] = 1; // 正向
            g[con[1]][con[0]] = 2; // 反向
        }
        dfs(g, 0, 2);

        return res;
    }

    public void dfs(int[][] g, int i, int dir) {
        if (vis[i]) return;
        vis[i] = true;
        if (dir == 1) res++;
        for (int j = 0; j < g.length; j++) {
            if (g[i][j] == 0) continue;
            dfs(g, j, g[i][j]);
        }
    }

//    private static int[][] dirs = new int[][] {
//            new int[] {1, 0}, new int[] {-1, 0},
//            new int[] {0, 1}, new int[] {0, -1}
//    };
    public int nearestExit(char[][] maze, int[] entrance) {
        Deque<int[]> q = new ArrayDeque<>();
        q.offer(entrance);
        maze[entrance[0]][entrance[1]] = '+';
        int m = maze.length, n = maze[0].length, step = 0;
        while (!q.isEmpty()) {
            step++;
            int l = q.size();
            for (int i = 0; i < l; i++) {
                int[] p = q.poll();
                for (int[] dir : dirs) {
                    int nx = p[0] + dir[0];
                    int ny = p[1] + dir[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == '.') {
                        if (nx == 0 || ny == 0 || nx == m - 1 || ny == n - 1)
                            return step;
                        q.offer(new int[] {nx, ny});
                        maze[nx][ny] = '+';
                    }
                }
            }
        }

        return -1;
    }

    public long totalCost(int[] costs, int k, int candidates) {
        Queue<Integer> q = new PriorityQueue<>((a, b) ->
                costs[a] == costs[b] ? a - b : costs[a] - costs[b]
        );
        int sum = 0, l = 0, r = costs.length - 1;
        while (l <= r && l < candidates) {
            q.offer(l++);
            if (l > r) break;
            q.offer(r--);
        }
        for (int i = 0; i < k; i++) {
            int idx = q.poll();
            if (idx < l && l <= r) q.offer(l++);
            else if (idx > r && l <= r) q.offer(r--);
            sum += costs[idx];
        }

        return sum;
    }

    public int longestStrChain(String[] words) {
        // 最长上升子序列
        Arrays.sort(words, (a, b) -> a.length() - b.length());
        int max = 1, n = words.length;
        // 以words[i]结尾的最长字符串链长度
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (isPredecessor(words[j], words[i])) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                    max = Math.max(max, dp[i]);
                }
            }
        }

        return max;
    }

    public boolean isPredecessor (String w1, String w2) {
        if (w1.length() != w2.length() - 1)
            return false;
        int i1 = 0, i2 = 0;
        char[] ch1 = w1.toCharArray(), ch2 = w2.toCharArray();
        while (i1 < ch1.length && i2 < ch2.length) {
            if (ch1[i1] == ch2[i2]) i1++;
            i2++;
        }
        return i1 == i2 - 1;
    }

    static class DinnerPlates {
        private final List<int[]> stacks = new ArrayList<>();
        private final int capacity;

        public DinnerPlates(int capacity) {
            this.capacity = capacity;
        }

        public void push(int val) {
            int len = stacks.size();
            for (int i = 0; i <= len; i++) {
                if (i == len)
                    stacks.add(new int[capacity + 1]);
                int[] stack = stacks.get(i);
                if (stack[0] < capacity) {
                    stack[++stack[0]] = val;
                    break;
                }
            }
        }

        public int pop() {
            for (int i = stacks.size() - 1; i >= 0; i--) {
                int[] stack = stacks.get(i);
                if (stack[0] == 0) continue;
                return stack[stack[0]--];
            }
            return -1;
        }

        public int popAtStack(int index) {
            if (index >= stacks.size()) return -1;
            int[] stack = stacks.get(index);
            if (stack[0] == 0) return -1;
            return stack[stack[0]--];
        }
    }

    public boolean equalFrequency(String word) {
        int[] map = new int[26];
        for (char c : word.toCharArray())
            map[c - 'a']++;
        Arrays.sort(map);
        int i = -1, pre = 0, n = word.length();
        while (++i < n && map[i] == 0);
        pre = map[i];
        while (++i < n && map[i] == pre);

        return i == n && map[i] == pre + 1;
    }

    public final static int[][] dirs = new int[][] {
            new int[]{1, 0, 2}, new int[]{0, 1, 4},
            new int[]{-1, 0, 6}, new int[]{0, -1, 8}
    };
    public int minPushBox(char[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] b = new int[2], s = new int[2];
        // 通过盒子和人的位置记录是否重复访问
        int[][] vis = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 'B') { b[0] = i; b[1] = j; grid[i][j] = '.'; }
                if (grid[i][j] == 'S') { s[0] = i; s[1] = j; grid[i][j] = '.'; }
            }
        }
        // bfs判断能否到达终点且能否到达箱子周围
        Deque<int[][]> q = new ArrayDeque<>();
        q.offer(new int[][]{b, s});
        int step = 0;
        while (!q.isEmpty()) {
            int l = q.size();
            for (int i = 0; i < l; i++) {
                int[][] p = q.poll();
                int bx = p[0][0], by = p[0][1], sx = p[1][0], sy = p[1][1];
                // 设置箱子位置
                grid[bx][by] = 'B';
                for (int[] dir : dirs) {
                    int bnx = bx + dir[0], bny = by + dir[1];
                    int spx = bx - dir[0], spy = by - dir[1];
                    if (bnx >= 0 && bnx < m && bny >= 0 && bny < n &&
                            (vis[bnx][bny] & (1 | dir[2])) == 0 &&
                            (grid[bnx][bny] == '.' || grid[bnx][bny] == 'T')) {
                        // 判断人能否走到箱子对侧位置
                        if (find(grid, sx, sy, spx, spy)) {
                            if (grid[bnx][bny] == 'T') return step + 1;
                            // 记录位置
                            q.offer(new int[][]{
                                    new int[]{bnx, bny},
                                    new int[]{bx, by}
                            });
                            vis[bnx][bny] |= (1 | dir[2]);
                        }
                    }
                }
                // 清除箱子位置
                grid[bx][by] = '.';
            }
            step++;
        }

        return -1;
    }

    public boolean find(char[][] g, int sx, int sy, int bpx, int bpy) {
        int m = g.length, n = g[0].length;
        boolean[][] vis = new boolean[m][n];
        Deque<int[]> q = new ArrayDeque<>();
        q.offer(new int[]{sx, sy});
        while (!q.isEmpty()) {
            int[] p = q.poll();
            sx = p[0]; sy = p[1];
            for (int[] dir : dirs) {
                int snx = sx + dir[0], sny = sy + dir[1];
                if (snx >= 0 && snx < m && sny >= 0 && sny < n &&
                        !vis[snx][sny] && (g[snx][sny] == '.' || g[snx][sny] == 'T')) {
                    if (snx == bpx && sny == bpy) return true;
                    vis[snx][sny] = true;
                    q.offer(new int[]{snx, sny});
                }
            }
        }
        return false;
    }

    public TreeNode sufficientSubset(TreeNode root, int limit) {
        TreeNode pNode = new TreeNode();
        pNode.left = root;
        dfs(pNode, root, 0, limit, true);
        return pNode.left;
    }

    public int dfs(TreeNode pNode, TreeNode node, int pre, int limit, boolean left) {
        pre += node.val;
        int lsum = Integer.MIN_VALUE, rsum = Integer.MIN_VALUE;

        if (node.left != null)
            lsum = dfs(node, node.left, pre, limit, true);
        if (node.right != null)
            rsum = dfs(node, node.right, pre, limit, false);

        if (lsum + pre < limit && rsum + pre < limit) {
            if (left)
                pNode.left = null;
            else
                pNode.right = null;
        }
        return Math.max(lsum, rsum) + node.val;
    }

    public int solution(int n, int[] array) {
        // Edit your code here
        // 单调栈记录能作为面积计算的元素
        int[] stack = new int[n];
        int sp = -1, max = 0;
        for (int i = 0; i < n; i++) {
            // 当前元素小于栈顶元素，则面积为，当前(元素下标-栈二元素下标) * 栈顶元素
            while (sp > -1 && array[stack[sp]] > array[i]) {
                int pre = sp == 0 ? -1 : stack[sp - 1];
                int len = i - 1 - pre;
                max = Math.max(max, len * array[stack[sp--]]);
            }
            stack[++sp] = i;
        }

        while (sp > -1) {
            int pre = sp == 0 ? -1 : stack[sp - 1];
            int len = n - 1 - pre;
            max = Math.max(max, len * array[stack[sp--]]);
        }

        return max;
    }

    public static int[][] dirs1 =
            new int[][] { new int[] {0, 1}, new int[] {1, 0}, new int[] {0, -1}, new int[] {-1, 0} };

    public int solution(int m, int n, int[][] a) {
        // PLEASE DO NOT MODIFY THE FUNCTION SIGNATURE
        // write code here
        int[][] max = new int[m][n];
        boolean[][] vis = new boolean[m][n];
        int res = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                int path = Math.max(
                        find(a, vis, max, res, i, j, 0, true),
                        find(a, vis, max, res, i, j, 0, false)
                );
                res = Math.max(res, path);
                max[i][j] = path;
            }
        return res;
    }

    public static int find (int[][] a, boolean[][] vis, int[][] max,
                            int curMax, int x, int y, int path, boolean up) {
        if (max[x][y] != 0 && max[x][y] + path <= curMax)
            return max[x][y];

        int m = a.length, n = a[0].length, maxPath = 0;
        vis[x][y] = true;
        for (int[] dir : dirs1) {
            int nx = x + dir[0]; int ny = y + dir[1];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && !vis[nx][ny] &&
                    a[nx][ny] != a[x][y] && a[nx][ny] > a[x][y] == up)
                maxPath = Math.max(maxPath, find(a, vis, max, curMax, x, y, path + 1, !up));
        }
        vis[x][y] = false;

        return maxPath + 1;
    }

    public int minFlipsMonoIncr(String s) {
        // dp0当前位置为0，保持有序最小反转次数；dp1当前位置为1，保持有有序最小反转次数
        int dp0 = 0, dp1 = 0;
        for (char c : s.toCharArray()) {
            // 当前位置为0
            if (c == '0') {
                // 反转当前位置
                dp1 = Math.min(dp0, dp1) + 1;
                dp0 = dp0;
            } else {
                dp1 = Math.min(dp0, dp1);
                dp0 = dp0 + 1;
            }
        }

        return Math.min(dp0, dp1);
    }

    public long largestSquareArea(int[][] bottomLeft, int[][] topRight) {
        // 两两查找矩阵，记录能放入最大正方形的矩阵交集
        int max = -1;
        for (int i = 0; i < bottomLeft.length; i++) {
            for (int j = i + 1; j < bottomLeft.length; j++) {
                // 矩阵相交面积
                int x1 = Math.max(bottomLeft[i][0], bottomLeft[j][0]);
                int y1 = Math.max(bottomLeft[i][1], bottomLeft[j][1]);
                int x2 = Math.min(topRight[i][0], topRight[j][0]);
                int y2 = Math.min(topRight[i][1], topRight[j][1]);

                int len = Math.min(x2 - x1, y2 - y1);
                if (len > 0)
                    max = Math.max(max, len * len);
            }
        }

        return max;
    }

    public static void main(String[] args) {
        Draft3 demo = new Draft3();
        long solution = demo.largestSquareArea(
                BuildArray.getDArray("[[1,1],[2,2],[3,1]]"),
                BuildArray.getDArray("[[3,3],[4,4],[6,6]]")
        );
        System.out.println(solution);
    }
}