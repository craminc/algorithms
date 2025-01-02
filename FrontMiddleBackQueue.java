public class FrontMiddleBackQueue {

    public class Node {
        public int val;
        public Node pre;
        public Node nex;
        public Node(int val) {
            this.val = val;
        }
    }

    public Node head;
    public Node tail;
    public Node mid;
    public int len;

    public FrontMiddleBackQueue() {
        head = new Node(-1);
        tail = new Node(-1);
        head.nex = tail;
        tail.pre = head;
        mid = tail;
    }

    public void pushFront(int val) {
        Node cur = new Node(val);
        cur.nex = head.nex;
        cur.pre = head;
        head.nex = cur;
        cur.nex.pre = cur;
        if ((++len & 1) == 1) mid = mid.pre;
    }

    public void pushMiddle(int val) {
        Node cur = new Node(val);
        cur.nex = mid;
        cur.pre = mid.pre;
        mid.pre = cur;
        cur.pre.nex = cur;
        if ((++len & 1) == 1) mid = mid.pre;
    }

    public void pushBack(int val) {
        Node cur = new Node(val);
        cur.nex = tail;
        cur.pre = tail.pre;
        tail.pre = cur;
        cur.pre.nex = cur;
        if ((++len & 1) == 0) mid = mid.nex;
    }

    public int popFront() {
        if (len == 0) return -1;
        Node cur = head.nex;
        head.nex = cur.nex;
        cur.nex.pre = head;
        if ((--len & 1) == 0) mid = mid.nex;
        return cur.val;
    }

    public int popMiddle() {
        if (len == 0) return -1;
        Node cur;
        if ((len & 1) == 1) cur = mid;
        else cur = mid.pre;
        cur.pre.nex = cur.nex;
        cur.nex.pre = cur.pre;
        if ((--len & 1) == 0) mid = mid.nex;
        return cur.val;
    }

    public int popBack() {
        if (len == 0) return -1;
        Node cur = tail.pre;
        tail.pre = cur.pre;
        cur.pre.nex = tail;
        if ((--len & 1) == 1) mid = mid.pre;
        return cur.val;
    }

    public static void main(String[] args) {
        FrontMiddleBackQueue q = new FrontMiddleBackQueue();
        q.pushBack(10);
        q.popMiddle();
    }
}

/**
 * Your FrontMiddleBackQueue object will be instantiated and called as such:
 * FrontMiddleBackQueue obj = new FrontMiddleBackQueue();
 * obj.pushFront(val);
 * obj.pushMiddle(val);
 * obj.pushBack(val);
 * int param_4 = obj.popFront();
 * int param_5 = obj.popMiddle();
 * int param_6 = obj.popBack();
 */