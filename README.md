<center>
    	<h1>算法精选</h1>
</center>



[TOC]

[哈希](#哈希)&emsp;&emsp;&emsp;&emsp;&emsp;[双指针](#双指针)&emsp;&emsp;&emsp;&emsp;&emsp;[滑动窗口](#滑动窗口)&emsp;&emsp;&emsp;&emsp;&emsp;[子串](#子串)&emsp;&emsp;&emsp;&emsp;&emsp;[数组](#数组)&emsp;&emsp;&emsp;&emsp;&emsp;[矩阵](#矩阵)&emsp;&emsp;&emsp;&emsp;&emsp;[链表](#链表)

[二叉树](#二叉树)&emsp;&emsp;&emsp;&emsp;[图论](#图论)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[回溯](#回溯)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[栈](#栈)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[堆](#堆)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[贪心](#贪心)&emsp;&emsp;&emsp;&emsp;&emsp;[其他](#其他)

------

## 哈希

### 两数之和

```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hash;
        for (int i = 0; i < nums.size(); i++) {
            auto it = hash.find(target - nums[i]);
            if (it != hash.end())
                return {it->second, i};
            hash[nums[i]] = i;
        }
        return {};
    }
};
```

### 字母异位词分组

```C++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ans;
        if (!strs.size())
            return ans;
        unordered_map<string, vector<string>> mp;
        for (const string& x : strs) {
            string k = x;
            sort(k.begin(), k.end());
            mp[k].emplace_back(x);
        }
        for (const auto& x : mp) {
            ans.emplace_back(x.second);
        }
        return ans;
    }
};
```

### 最长连续序列

```C++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> set;
        for (const int& x : nums) {
            set.emplace(x);
        }
        int last = 0;
        for (const int& x : set) {
            if (!set.count(x - 1)) {
                int current = x;
                int currentLast = 1;
                while (set.count(current + 1)) {
                    ++current;
                    ++currentLast;
                }
                last = max(last, currentLast);
            }
        }
        return last;
    }
};
```

## 双指针

### 移动零

```C++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int n = nums.size(), i = 0, j = 0;
        while (j < n) {
            if (nums[j]) {
                swap(nums[j], nums[i]);
                ++i;
            }
            ++j;
        }
    }
};
```

### 盛最多水的容器

```C++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size() - 1, maxH = 0;
        while (i < j) {
            maxH = max(maxH, min(height[i], height[j]) * (j - i));
            if (height[i] <= height[j])
                ++i;
            else
                --j;
        }
        return maxH;
    }
};
```

### 三数之和

```C++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            if (nums[i] > 0)
                return ans;
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            int des = -nums[i], k = n - 1;
            for (int j = i + 1; j < n; ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1])
                    continue;
                while (j < k && nums[k] + nums[j] > des)
                    --k;
                if (j == k)
                    break;
                if (nums[k] + nums[j] == des) {
                    ans.push_back({nums[i], nums[j], nums[k]});
                }
            }
        }
        return ans;
    }
};
```

### 接雨水

```C++
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0, i = 0, j = height.size() - 1;
        int iMax = 0, jMax = 0;
        while (i <= j) {
            iMax = max(iMax, height[i]);
            jMax = max(jMax, height[j]);
            if (iMax <= jMax) {
                ans += iMax - height[i];
                ++i;
            } else {
                ans += jMax - height[j];
                --j;
            }
        }
        return ans;
    }
};
```

## 滑动窗口

### 无重复字符的最长子串

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> set;
        int ans = 0, n = s.size(), j = -1;
        for (int i = 0; i < n; ++i) {
            if (i)
                set.erase(s[i - 1]);
            while (j + 1 < n && !set.count(s[j + 1])) {
                set.emplace(s[j + 1]);
                ++j;
            }
            ans = max(ans, j - i + 1);
        }
        return ans;
    }
};
```

### 找到字符串中所有字母异位词

```C++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int n1 = s.size(), n2 = p.size();
        if (n1 < n2)
            return {};
        vector<int> ans;
        vector<int> cnt(26);
        // p,s凹凸计数，建立窗口
        for (int i = 0; i < n2; ++i) {
            ++cnt[s[i] - 'a'];
            --cnt[p[i] - 'a'];
        }
        // 记录绝对值总数：1多   0适   -1少
        int diff = 0;
        for (int i = 0; i < 26; ++i)
            if (cnt[i])
                ++diff;
        if (!diff)
            ans.emplace_back(0);
        // 滑动窗口
        for (int i = 0; i < n1 - n2; ++i) {
            // 左边界
            if (cnt[s[i] - 'a'] == 0)
                ++diff;
            else if (cnt[s[i] - 'a'] == 1)
                --diff;
            --cnt[s[i] - 'a'];
            // 右边界
            if (cnt[s[i + n2] - 'a'] == 0)
                ++diff;
            else if (cnt[s[i + n2] - 'a'] == -1)
                --diff;
            ++cnt[s[i + n2] - 'a'];

            if (!diff)
                ans.emplace_back(i + 1);
        }
        return ans;
    }
};
```

## 子串

### 和为K的子数组

```C++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int cnt = 0, pre = 0;
        unordered_map<int, int> mp;
        mp[0] = 1; //<Sn,f>
        for (const int& x : nums) {
            pre += x;
            if (mp.count(pre - k))
                cnt += mp[pre - k];
            ++mp[pre];
        }
        return cnt;
    }
};
```

### 滑动窗口最大值

```C++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        deque<int> q;
        // 创建初始窗口，严格递减队列
        for (int i = 0; i < k; ++i) {
            while (!q.empty() && nums[i] >= nums[q.back()])
                q.pop_back();
            q.push_back(i);
        }
        // 当前队头最大
        vector<int> ans;
        ans.emplace_back(nums[q.front()]);
        // 滑动窗口
        for (int i = k; i < n; ++i) {
            while (!q.empty() && nums[i] >= nums[q.back()])
                q.pop_back();
            q.push_back(i);
            // 保证队首有效性
            while (q.front() <= i - k)
                q.pop_front();
            // 队头最大
            ans.emplace_back(nums[q.front()]);
        }
        return ans;
    }
};
```

### 最小覆盖子串

```C++
class Solution {
public:
    string minWindow(string s, string t) {
        int n = s.size(), cat = 0;
        int S[58]{}, T[58]{};
        // 记录t中字符种数
        for (const char& x : t) {
            if (!T[x - 'A'])
                ++cat;
            ++T[x - 'A'];
        }
        // 窗口右扩
        int L = -1, R = n, i = 0;
        for (int j = 0; j < n; ++j) {
            ++S[s[j] - 'A'];
            if (S[s[j] - 'A'] == T[s[j] - 'A'])
                --cat;
            // 窗口左缩
            while (!cat) {
                if (j - i < R - L) {
                    L = i;
                    R = j;
                }
                if (S[s[i] - 'A'] == T[s[i] - 'A'])
                    ++cat;
                --S[s[i] - 'A'];
                ++i;
            }
        }
        return L < 0 ? "" : s.substr(L, R - L + 1);
    }
};
```

## 数组

### 最大子数组和

```C++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int seq = 0, ans = nums[0];
        for (const int& x : nums) {
            seq = max(seq + x, x);
            ans = max(ans, seq);
        }
        return ans;
    }
};
```

### 合并区间

```C++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());

        vector<vector<int>> ans;
        int m = intervals.size();

        for (int i = 0; i < m; ++i) {
            if (ans.empty() || ans.back()[1] < intervals[i][0])
                ans.push_back({intervals[i][0], intervals[i][1]});
            else
                ans.back()[1] = max(ans.back()[1], intervals[i][1]);
        }
        return ans;
    }
};
```

### 轮转数组

```C++
class Solution {
    void reverse(vector<int>& nums, int i, int j) {
        while (i < j) {
            swap(nums[i], nums[j]);
            ++i;
            --j;
        }
    }

public:
    void rotate(vector<int>& nums, int k) {
        k %= nums.size();
        reverse(nums, 0, nums.size() - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.size() - 1);
    }
};
```

### 除自身以外数组的乘积

```C++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> v(n, 1);
        for (int i = 1; i < n; ++i)
            v[i] = v[i - 1] * nums[i - 1];
        int pre = nums[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            v[i] *= pre;
            pre *= nums[i];
        }
        return v;
    }
};
```

### 缺失的第一个正数

```C++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        // 对[1,n]中的值恢复其映射
        for (int i = 0; i < n; ++i) {
            // 为防止重复元素造成死循环，需要以置换位置的两元素判断而不是判断映射
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i])
                swap(nums[i], nums[nums[i] - 1]);
        }
        for (int i = 0; i < n; ++i)
            if (nums[i] != i + 1)
                return i + 1;
        return n + 1;
    }
};
```

## 矩阵

### 矩阵置零

```C++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        bool flag = false;
        for (int i = 0; i < m; ++i) {
            if (!matrix[i][0])
                flag = true;
            for (int j = 1; j < n; ++j) {
                if (!matrix[i][j])
                    matrix[i][0] = matrix[0][j] = 0;
            }
        }

        for (int i = m - 1; i >= 0; --i) {
            for (int j = 1; j < n; ++j) {
                if (!matrix[i][0] || !matrix[0][j])
                    matrix[i][j] = 0;
            }
            if (flag)
                matrix[i][0] = 0;
        }
    }
};
```

### 螺旋矩阵

```C++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> ans;
        int u = 0, d = matrix.size() - 1, l = 0, r = matrix[0].size() - 1;
        // 当相对的边界发生交错，遍历完毕
        while (true) 
        {
            for (int i = l; i <= r; ++i)    ans.emplace_back(matrix[u][i]);
            if (++u > d)    break;
            for (int i = u; i <= d; ++i)    ans.emplace_back(matrix[i][r]);
            if (--r < l)    break;
            for (int i = r; i >= l; --i)    ans.emplace_back(matrix[d][i]);
            if (--d < u)    break;
            for (int i = d; i >= u; --i)    ans.emplace_back(matrix[i][l]);
            if (++l > r)    break;
        }
        return ans;
    }
};
```

### 旋转图像

```C++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n / 2; ++i)
            for (int j = 0; j < n; ++j)
                swap(matrix[i][j], matrix[n - i - 1][j]);
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < i; ++j)
                swap(matrix[i][j], matrix[j][i]);
    }
};
```

### 搜索二维矩阵II

```C++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (matrix[i][j] == target)
                return true;
            if (matrix[i][j] > target)
                --j;
            else
                ++i;
        }
        return false;
    }
};
```

## 链表

### 相交链表

```C++
class Solution {
public:
    ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
        ListNode *p = headA, *q = headB;
        while (p != q) {
            p = p ? p->next : headB;
            q = q ? q->next : headA;
        }
        return p;
    }
};
```

### 反转链表

```C++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *P = nullptr, *C = head;
        while (C) {
            ListNode* N = C->next;
            C->next = P;
            P = C;
            C = N;
        }
        return P;
    }
};
```

### 回文链表

```C++
class Solution {
public:
    bool isPalindrome(ListNode* head) {   //空间O(1)：快慢指针定位，反转后半段，扫描后恢复
        vector<int> v;
        ListNode* p = head;
        while (p) {
            v.emplace_back(p->val);
            p = p->next;
        }
        int i = 0, j = v.size() - 1;
        while (i < j) {
            if (v[i] != v[j])
                return false;
            ++i;
            --j;
        }
        return true;
    }
};
```

### 环形链表

```C++
class Solution {
public:
    ListNode* detectCycle(ListNode* head) {
        ListNode *p = head, *q = head;
        while (q) {
            if (!q->next)   return nullptr;
            q = q->next->next;
            p = p->next;
            if (p == q) {
                q = head;
                while (p != q) {
                    q = q->next;
                    p = p->next;
                }
                return p;
            }
        }
        return nullptr;
    }
};
```

### 合并两个有序链表

```C++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* L, ListNode* R) {
        if (!L || !R)   return L ? L : R;

        ListNode *dummy = new ListNode(-1), *p = dummy;
        while (L && R) {
            if (L->val <= R->val) {
                p->next = L;
                L = L->next;
            } else {
                p->next = R;
                R = R->next;
            }
            p = p->next;
        }
        p->next = L ? L : R;

        p = dummy->next;delete dummy;
        return p;
    }
};
```

### 两数相加

```C++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *H = nullptr, *T = nullptr;
        bool carry = false;
        while (l1 || l2) {
            int n1 = l1 ? l1->val : 0;
            int n2 = l2 ? l2->val : 0;
            int sum = n1 + n2 + carry;

            if (!H) {
                H = T = new ListNode(sum % 10);
            } else {
                T->next = new ListNode(sum % 10);
                T = T->next;
            }
            carry = sum >= 10;
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }
        if (carry)  T->next = new ListNode(1, nullptr);  
        return H;
    }
};
```

### 删除链表的倒数第N个结点

```C++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(-1, head);
        ListNode *p = dummy, *q = head;
        for (int i = 0; i < n; ++i)
            q = q->next;
        while (q) {
            q = q->next;
            p = p->next;
        }
        p->next = p->next->next;

        q = dummy->next;delete dummy;
        return q;
    }
};
```

### 两两交换链表中的结点

```C++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode* dummy = new ListNode(-1, head);
        ListNode* current = dummy;
        while (current->next && current->next->next) {
            ListNode* p = current->next;
            ListNode* q = current->next->next;
            p->next = q->next;
            q->next = p;
            current->next = q;
            current = p;
        }

        current = dummy->next;delete dummy;
        return current;
    }
};
```

### K个一组反转链表

```C++
class Solution {
    // 翻转链表：反转指针，逆序返回
    pair<ListNode*, ListNode*> RVS(ListNode* head, ListNode* tail) {
        ListNode *P = nullptr, *C = head;
        tail = tail->next;
        while (C != tail) {
            ListNode* N = C->next;
            C->next = P;
            P = C;
            C = N;
        }
        return {P, head};
    }

public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* dummy = new ListNode(-1, head);
        ListNode* pre = dummy;

        while (head) {
            ListNode* tail = pre;
            for (int i = 0; i < k; ++i) {
                tail = tail->next;
                if (!tail) {
                    pre = dummy->next;delete dummy;
                    return pre;
                }
            }
            ListNode* tailNext = tail->next;
            tie(head, tail) = RVS(head, tail);
            // 连接前链
            pre->next = head;
            tail->next = tailNext;
            pre = tail;
            head = tail->next;
        }

        pre = dummy->next;delete dummy;
        return pre;
    }
};
```

### 随机链表的复制

```C++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head)  return nullptr;
        // 添加后继A'
        for (Node* N = head; N; N = N->next->next) {
            Node* A = new Node(N->val);
            A->next = N->next;
            N->next = A;
        }
        // 完善随机指针
        for (Node* N = head; N; N = N->next->next) {
            Node* R = N->random ? N->random->next : nullptr;
            N->next->random = R;
        }
        // 串联A'
        Node* p = head->next;
        for (Node* N = head; N; N = N->next) {
            Node* q = N->next;
            N->next = N->next->next;
            q->next = N->next ? N->next->next : nullptr;
        }

        return p;
    }
};
```

### 排序链表

```C++
class Solution {
    ListNode* merge(ListNode* L, ListNode* R) {
        if (!L || !R)
            return L ? L : R;

        ListNode *dummy = new ListNode(-1), *p = dummy;
        while (L && R) {
            if (L->val <= R->val) {
                p->next = L;
                L = L->next;
            } else {
                p->next = R;
                R = R->next;
            }
            p = p->next;
        }
        p->next = L ? L : R;

        p = dummy->next;
        delete dummy;
        return p;
    }

public:
    // 自底向上的归并排序
    ListNode* sortList(ListNode* head) {
        if (!head)  return nullptr;
        ListNode* dummy = new ListNode(-1, head);
        int len = 0;
        // 计算链表长度
        while (head) {
            ++len;
            head = head->next;
        }
        // 维护子链大小
        for (int sublen = 1; sublen < len; sublen <<= 1) {
            ListNode *pre = dummy, *p = dummy->next;
            // 归并若干个子链
            while (p) {
                // 划分第一个子链
                ListNode* H1 = p;
                for (int i = 1; i < sublen && p->next; ++i)
                    p = p->next;
                ListNode* pNext = p->next;
                p->next = nullptr;
                p = pNext;
                // 划分第二个子链
                ListNode* H2 = p;
                for (int i = 1; i < sublen && p && p->next; ++i)
                    p = p->next;
                pNext = nullptr;
                if (p) {
                    pNext = p->next;
                    p->next = nullptr;
                }
                // 归并两条链，串入总链
                pre->next = merge(H1, H2);
                while (pre->next) {
                    pre = pre->next;
                }
                p = pNext;
            }
        }
        head = dummy->next;delete dummy;
        return head;
    }
};
```

### 合并K个升序链表

```C++
class Solution {
    ListNode* merge2(ListNode* L, ListNode* R) {
        if (!L || !R)   return L ? L : R;
        ListNode *dummy = new ListNode(-1), *p = dummy;
        while (L && R) {
            if (L->val <= R->val) {
                p->next = L;
                L = L->next;
            } else {
                p->next = R;
                R = R->next;
            }
            p = p->next;
        }
        p->next = L ? L : R;
        p = dummy->next;delete dummy;
        return p;
    }
    ListNode* merge(vector<ListNode*>& lists, int l, int r) {
        if (l == r) return lists[l];
        if (l > r)  return nullptr;
        int mid = (l + r) >> 1;
        return merge2(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return merge(lists, 0, lists.size() - 1);
    }
};
```

### LRU缓存

```C++
struct DlinkList {
    int key;
    int val;
    DlinkList* next;
    DlinkList* pre;
    DlinkList() : key(0), val(0), next(nullptr), pre(nullptr){};
    DlinkList(int k, int v) : key(k), val(v), next(nullptr), pre(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DlinkList*> Cache;
    DlinkList* head;
    DlinkList* tail;
    int size;
    int capacity;

public:
    LRUCache(int cp) : size(0), capacity(cp) {
        head = new DlinkList();
        tail = new DlinkList();
        head->next = tail;
        tail->pre = head;
    }

    int get(int key) {
        if (Cache.count(key)) {
            DlinkList* node = Cache[key];
            moveToHead(node);
            return node->val;
        }
        return -1;
    }

    void put(int key, int value) {
        if (Cache.count(key)) {
            DlinkList* node = Cache[key];
            node->val = value;
            moveToHead(node);
        } else {
            DlinkList* node = new DlinkList(key, value);
            addToHead(node);
            Cache[key] = node;
            ++size;
            if (size > capacity) {
                int k = removeTail();
                Cache.erase(k);
                --size;
            }
        }
    }

    void addToHead(DlinkList* node) {
        node->next = head->next;
        node->pre = head;
        head->next->pre = node;
        head->next = node;
    }

    void removeNode(DlinkList* node) {
        node->pre->next = node->next;
        node->next->pre = node->pre;
    }
    // 移至头部
    void moveToHead(DlinkList* node) {
        removeNode(node);
        addToHead(node);
    }
    // 移除尾部
    int removeTail() {
        DlinkList* node = tail->pre;
        tail->pre = node->pre;
        node->pre->next = tail;
        int k = node->key;
        delete node;
        return k;
    }
};
```

## 二叉树

### 二叉树的中序遍历

```C++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> v;
        TreeNode* pre;
        while (root) {
            if (root->left) {
                pre = root->left;
                while (pre->right && pre->right != root) {
                    pre = pre->right;
                }
                // 线索化
                if (!pre->right) {
                    pre->right = root;
                    root = root->left;
                } else { // 去线索化
                    pre->right = nullptr;
                    v.emplace_back(root->val);
                    root = root->right;
                }
            } else {
                v.emplace_back(root->val);
                root = root->right;
            }
        }
        return v;
    }
};
```

### 二叉树的最大深度

```C++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        return root ? max(maxDepth(root->left), maxDepth(root->right)) + 1 : 0;
    }
};
```

### 反转二叉树

```C++
class Solution {
    void dfs(TreeNode* root) {
        if (!root)  return;
        swap(root->left, root->right);
        dfs(root->right);
        dfs(root->left);
    }

public:
    TreeNode* invertTree(TreeNode* root) {
        dfs(root);
        return root;
    }
};
```

### 对称二叉树

```C++
class Solution {
    bool dfs(TreeNode* p, TreeNode* q) {
        if (!p && !q)   return true;
        if (!p || !q)   return false;
        return p->val == q->val && dfs(p->left, q->right) && dfs(p->right, q->left);
    }

public:
    bool isSymmetric(TreeNode* root) { return dfs(root, root); }
};
```

### 二叉树的直径

```C++
class Solution {
    int maxEdge = 0;
    int dfs(TreeNode* root) {
        if (!root)  return 0;
        int L = dfs(root->left);
        int R = dfs(root->right);
        maxEdge = max(maxEdge, L + R);
        return max(L, R) + 1;
    }

public:
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return maxEdge;
    }
};
```

### 二叉树的层序遍历

```C++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (!root)  return ans;
        queue<TreeNode*> q;
        q.emplace(root);
        while (!q.empty()) {
            int size = q.size();
            TreeNode* node;
            vector<int> v;
            while (size) {
                node = q.front();
                v.emplace_back(node->val);
                if (node->left)
                    q.emplace(node->left);
                if (node->right)
                    q.emplace(node->right);
                q.pop();
                --size;
            }
            ans.emplace_back(v);
        }
        return ans;
    }
};
```

### 将有序数组转化为二叉查找树

```C++
class Solution {
    TreeNode* bst(vector<int>& nums, int L, int R) {
        if (L > R)  return nullptr;
        int mid = (L + R) / 2;
        TreeNode* node = new TreeNode(nums[mid]);
        node->left = bst(nums, L, mid - 1);
        node->right = bst(nums, mid + 1, R);
        return node;
    }

public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return bst(nums, 0, nums.size() - 1);
    }
};
```

### 验证二叉查找树

```C++
class Solution {
    bool bst(TreeNode* r, long long L, long long R) {
        if (!r)
            return true;
        if (r->val <= L || r->val >= R)
            return false;
        return bst(r->left, L, r->val) && bst(r->right, r->val, R);
    }
    // Windows下需要用LLONG_MIN,LLONG_MAX
public:
    bool isValidBST(TreeNode* root) { return bst(root, LONG_MIN, LONG_MAX); }
};
```

### 二叉查找树中第K小的元素

```C++
class Solution {
    int value, key;
    void inOrder(TreeNode* root) {
        if (!root)      return;
        inOrder(root->left);
        if (--key == 0)     value = root->val;
        inOrder(root->right);
    }

public:
    int kthSmallest(TreeNode* root, int k) {
        key = k;
        inOrder(root);
        return value;
    }
};
```

### 二叉树的右视图

```C++
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        unordered_map<int, int> mp;
        int maxDep = -1;

        stack<TreeNode*> sNode;
        stack<int> sDep;
        sNode.push(root);
        sDep.push(0);

        while (!sNode.empty()) {
            TreeNode* node = sNode.top();sNode.pop();
            int depth = sDep.top();sDep.pop();

            if (node) {
                maxDep = max(maxDep, depth);
                if (!mp.count(depth))
                    mp[depth] = node->val;

                sNode.push(node->left);
                sNode.push(node->right);
                sDep.push(depth + 1);
                sDep.push(depth + 1);
            }
        }

        vector<int> R;
        for (int i = 0; i <= maxDep; ++i)
            R.emplace_back(mp[i]);
        return R;
    }
};
```

### 二叉树展开为链表

```C++
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* p = root;
        while (p) {
            if (p->left) {
                // 找到前驱
                TreeNode* pre = p->left;
                while (pre->right)
                    pre = pre->right;
                // 根——>左——>右
                pre->right = p->right;
                p->right = p->left;
                p->left = nullptr;
            }
            p = p->right;
        }
    }
};
```

### 从先序与中序序列构造二叉树

```C++
class Solution {
    unordered_map<int, int> inId;
    TreeNode* DFS(const vector<int>& preorder, const vector<int>& inorder, int preL, int preR, int inL, int inR) {
        if (preL > preR)
            return nullptr;
        int preRoot = preL;
        int inRoot = inId[preorder[preRoot]];

        TreeNode* root = new TreeNode(preorder[preRoot]);
        int len_L = inRoot - inL;
        root->left = DFS(preorder, inorder, preL + 1, preL + len_L, inL, inRoot - 1);
        root->right = DFS(preorder, inorder, preL + len_L + 1, preR, inRoot + 1, inR);
        return root;
    }

public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = inorder.size();
        for (int i = 0; i < n; ++i)
            inId[inorder[i]] = i;
        return DFS(preorder, inorder, 0, n - 1, 0, n - 1);
    }
};
```

### 路径总和III

```C++
class Solution {
    unordered_map<long long, int> mp;
    int dfs(TreeNode* root, long long pre, int k) {
        if (!root)  return 0;

        int ctn = 0;
        pre += root->val;
        if (mp.count(pre - k))
            ctn = mp[pre - k];

        ++mp[pre];
        ctn += dfs(root->left, pre, k);
        ctn += dfs(root->right, pre, k);
        --mp[pre];
        
        return ctn;
    }

public:
    int pathSum(TreeNode* root, int targetSum) {
        mp[0] = 1;
        return dfs(root, 0, targetSum);
    }
};
```

### 二叉树的最近公共祖先

```C++
class Solution {
    TreeNode* ans;
    bool dfs(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root)      return false;
        bool L = dfs(root->left, p, q);
        bool R = dfs(root->right, p, q);
        if (L && R || ((root->val == p->val || root->val == q->val) && (L || R)))
            ans = root;
        return L || R || root->val == p->val || root->val == q->val;
    }

public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        dfs(root, p, q);
        return ans;
    }
};
```

### 二叉树中的最大路径和

```C++
class Solution {
    int maxSum = INT_MIN;
    int subMax(TreeNode* root) {
        if (!root)      return 0;
        int L = max(subMax(root->left), 0);
        int R = max(subMax(root->right), 0);
        int sum = root->val + L + R;
        maxSum = max(maxSum, sum);
        return root->val + max(L, R);
    }

public:
    int maxPathSum(TreeNode* root) {
        subMax(root);
        return maxSum;
    }
};
```

## 图论

### 岛屿数量

```C++
class Solution {
    bool inside(vector<vector<char>>& grid, int i, int j) {
        return i >= 0 && i < grid.size() && j >= 0 && j < grid[0].size();
    }
    void DFS(vector<vector<char>>& grid, int i, int j) {
        if (!inside(grid, i, j) || grid[i][j] != '1')       return;
        grid[i][j] = '2';
        //上下左右
        DFS(grid, i - 1, j);
        DFS(grid, i + 1, j);
        DFS(grid, i, j - 1);
        DFS(grid, i, j + 1);
    }

public:
    int numIslands(vector<vector<char>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int cnt = 0;

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j] == '1') {
                    DFS(grid, i, j);
                    ++cnt;
                }
        return cnt;
    }
};
```

### 烂橘子

```C++
class Solution {
    int org;
    int rot[10][10];
    int X[4] = {-1, 1, 0, 0};
    int Y[4] = {0, 0, -1, 1};

public:
    int orangesRotting(vector<vector<int>>& grid) {

        queue<pair<int, int>> Q;
        memset(rot, -1, sizeof(rot));
        int m = grid.size(), n = grid[0].size(), minu = 0;
        org = 0;
        // 初始化烂橘子
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 2) {
                    Q.emplace(i, j);
                    rot[i][j] = 0;
                } else if (grid[i][j] == 1)
                    ++org;
            }

        while (!Q.empty()) {
            auto [r, c] = Q.front();Q.pop();

            for (int i = 0; i < 4; ++i) {
                int dx = r + X[i];
                int dy = c + Y[i];
                // 越界，空格，烂橘子
                if (dx < 0 || dx >= m || dy < 0 || dy >= n ||
                    rot[dx][dy] != -1 || !grid[dx][dy])
                    continue;
                // 入队
                Q.emplace(dx, dy);
                // 时间推移
                rot[dx][dy] = rot[r][c] + 1;

                minu = rot[dx][dy];
                --org;
                if (!org)   break;
            }
        }
        return org ? -1 : minu;
    }
};
```

### 课程表

```C++
class Solution {
    bool valid = true;
    vector<int> visited;
    vector<vector<int>> arc;

    void dfs(int u) {
        // 搜索中
        visited[u] = 1;
        for (const int& v : arc[u]) {
            if (!visited[v]) {
                dfs(v);
                if (!valid) return;
            } else if (visited[v] == 1) {
                valid = false;
                return;
            }
        }
        // 已完成
        visited[u] = 2;
    }

public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        arc.resize(numCourses);
        //未搜索
        visited.resize(numCourses);
        //邻接表
        for (const auto& x : prerequisites) {
            arc[x[1]].emplace_back(x[0]);
        }
        for (int i = 0; i < numCourses && valid; ++i) {
            if (!visited[i])
                dfs(i);
        }
        return valid;
    }
};
```

### 实现 Trie（前缀树）

```C++
class Trie { // 26叉树
    bool isEnd;
    vector<Trie*> child;

    Trie* searchWd(string word) {
        Trie* node = this;
        for (char x : word) {
            x -= 'a';
            if (!node->child[x])
                return nullptr;
            node = node->child[x];
        }
        return node;
    }

public:
    Trie() : isEnd(false), child(26) {}
    // 插入串
    void insert(string word) {
        Trie* node = this;
        for (char x : word) {
            x -= 'a';
            if (!node->child[x])
                node->child[x] = new Trie();
            node = node->child[x];
        }
        node->isEnd = true;
    }
    // 查找串
    bool search(string word) {
        Trie* node = searchWd(word);
        return node && node->isEnd;
    }
    // 查找子串
    bool startsWith(string prefix) { return searchWd(prefix); }
};
```

## 回溯

### 全排列

```C++
class Solution {
    void backtrack(vector<vector<int>>& nums, vector<int>& v, int pre, int end) {
        // 递归到底了，装填
        if (pre == end) {
            nums.emplace_back(v);
            return;
        }
        for (int i = pre; i < end; ++i) {
            // pre之前的序列表示已填
            swap(v[pre], v[i]);
            backtrack(nums, v, pre + 1, end);
            swap(v[pre], v[i]);
        }
    }

public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> v;
        backtrack(v, nums, 0, nums.size());
        return v;
    }
};
```

### 子集

```C++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> ans;
        // n个元素，代表n位，2^n种状态
        for (int i = 0; i < 1 << n; ++i) {
            vector<int> v;
            for (int j = 0; j < n; ++j) {
                // 与到说明此位对应元素
                if (i & 1 << j)
                    v.emplace_back(nums[j]);
            }
            ans.emplace_back(v);
        }
        return ans;
    }
};
```

### 电话号码的字母组合

```C++
class Solution {
    unordered_map<char, string> dic{{'2', "abc"}, {'3', "def"}, {'4', "ghi"},
                                    {'5', "jkl"}, {'6', "mno"}, {'7', "pqrs"},
                                    {'8', "tuv"}, {'9', "wxyz"}};
    void backtrack(vector<string>& wd, string& word, string& str, int bg,
                   int len) {
        if (bg == len) {
            wd.emplace_back(str);
            return;
        }
        for (const char& x : dic[word[bg]]) {
            str.push_back(x);
            backtrack(wd, word, str, bg + 1, len);
            str.pop_back();
        }
    }

public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty())     return {};
        string str;
        vector<string> wd;
        backtrack(wd, digits, str, 0, digits.size());
        return wd;
    }
};
```

### 组合总和

```C++
class Solution {
    void backtrack(vector<vector<int>>& subsets, vector<int>& candi,
                   vector<int>& v, int target, int start) {
        if (!target) {
            subsets.emplace_back(v);
            return;
        }
        int n = candi.size();
        // 剪枝：升序规避重复子集
        for (int i = start; i < n; ++i) {
            if (target < candi[i])
                return;
            // 尝试当前
            v.emplace_back(candi[i]);
            backtrack(subsets, candi, v, target - candi[i], i);
            v.pop_back();
        }
    }

public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> subsets;
        vector<int> v;
        sort(candidates.begin(), candidates.end());
        backtrack(subsets, candidates, v, target, 0);
        return subsets;
    }
};
```

### 括号生成

```C++
class Solution {
    void backtrack(vector<string>& v, string& str, int open, int close, int n) {
        if (str.size() == 2 * n) {
            v.emplace_back(str);
            return;
        }
        // 先放左括号
        if (open < n) {
            str.push_back('(');
            backtrack(v, str, open + 1, close, n);
            str.pop_back();
        }
        // 能放右括号
        if (close < open) {
            str.push_back(')');
            backtrack(v, str, open, close + 1, n);
            str.pop_back();
        }
    }

public:
    vector<string> generateParenthesis(int n) {
        vector<string> v;
        string str;
        backtrack(v, str, 0, 0, n);
        return v;
    }
};
```

### 单词搜索

```C++
class Solution {
    bool dfs(vector<vector<char>>& board, string word, int k, int i, int j,
             int m, int n) {
        if (k == word.size()) {
            return true;
        }
        if (i < 0 || i >= m || j < 0 || j >= n || word[k] != board[i][j])
            return false;
        board[i][j] = '\0';
        bool ret = dfs(board, word, k + 1, i - 1, j, m, n) ||
                   dfs(board, word, k + 1, i + 1, j, m, n) ||
                   dfs(board, word, k + 1, i, j - 1, m, n) ||
                   dfs(board, word, k + 1, i, j + 1, m, n);
        board[i][j] = word[k];

        return ret;
    }

public:
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size(), n = board[0].size();
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (dfs(board, word, 0, i, j, m, n))
                    return true;
        return false;
    }
};
```

### 分割回文串

```C++
class Solution {
    void backtrack(vector<vector<string>>& ret, vector<vector<bool>>& dp,
                   vector<string>& v, const string& s, int start, int n) {
        if (start == n) {
            ret.emplace_back(v);
            return;
        }

        for (int i = start; i < n; ++i) {
            // 如果是回文
            if (dp[start][i]) {
                v.emplace_back(s.substr(start, i - start + 1));
                backtrack(ret, dp, v, s, i + 1, n);
                v.pop_back();
            }
        }
    }

public:
    vector<vector<string>> partition(string s) {
        int n = s.size();
        vector<string> v;
        vector<vector<string>> ret;
        vector<vector<bool>> dp(n, vector<bool>(n, true));

        for (int i = n; i >= 0; --i)
            for (int j = i + 1; j < n; ++j)
                dp[i][j] = (s[i] == s[j]) && dp[i + 1][j - 1];

        backtrack(ret, dp, v, s, 0, n);
        return ret;
    }
};
```

### N皇后

```C++
class Solution {
    vector<string> generateQ(const vector<int>& Q, int n) {
        vector<string> v;
        // Q表征每行中皇后出现的列索引
        for (int i = 0; i < n; ++i) {
            string str(n, '.');
            str[Q[i]] = 'Q';
            v.emplace_back(str);
        }
        return v;
    }

    void backtrack(vector<vector<string>>& ans, vector<int>& Q,
                   unordered_set<int>& cols, unordered_set<int>& diag1,
                   unordered_set<int>& diag2, int row, int n) {
        // 已满，装填
        if (row == n) {
            vector<string> str = generateQ(Q, n);
            ans.emplace_back(str);
            return;
        }
        // 扫描当前行的每列
        for (int i = 0; i < n; ++i) {
            if (cols.count(i))
                continue;
            // 正斜线：行 - 列
            int d1 = row - i;
            if (diag1.count(d1))
                continue;
            // 反斜线：行 + 列
            int d2 = row + i;
            if (diag2.count(d2))
                continue;

            Q[row] = i;
            cols.emplace(i);
            diag1.emplace(d1);
            diag2.emplace(d2);
            backtrack(ans, Q, cols, diag1, diag2, row + 1, n);
            Q[row] = -1;
            cols.erase(i);
            diag1.erase(d1);
            diag2.erase(d2);
        }
    }

public:
    vector<vector<string>> solveNQueens(int n) {
        vector<int> Q(n, -1);
        vector<vector<string>> ans;
        unordered_set<int> cols;
        unordered_set<int> diag1;
        unordered_set<int> diag2;
        backtrack(ans, Q, cols, diag1, diag2, 0, n);
        return ans;
    }
};
```

## 二分查找

### 搜索插入位置

```C++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int lo = 0, hi = nums.size() - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (nums[mid] == target)
                return mid;
            else if (nums[mid] < target)
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        return lo;
    }
};
```

### 在排序数组中查找元素的第一个和最后一个位置

```C++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        auto bg = lower_bound(nums.begin(), nums.end(), target);
        auto ed = upper_bound(nums.begin(), nums.end(), target);
        if (bg == nums.end() || *bg != target)      return {-1, -1};
        return {(int)(bg - nums.begin()), (int)(ed - nums.begin()) - 1};
    }
};
```

### 搜索旋转排序数组

```C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int lo = 0, hi = nums.size() - 1;
        // 相遇即找到
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            // [0,x,mid] [0,x,rota,mid] [0,rota,x,mid]向前规约，否则向后
            if (nums[mid] < nums[0] ^ target <= nums[mid] ^ target >= nums[0])
                lo = mid + 1;
            else
                hi = mid;
        }
        return nums[lo] == target ? lo : -1;
    }
};
```

### 寻找旋转排序数组中的最小值

```C++
class Solution {
public:
    int findMin(vector<int>& nums) {
        int lo = 0, hi = nums.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            // 对比中点和尾部，判断规约方向
            if (nums[mid] > nums[hi])
                lo = mid + 1;
            else
                hi = mid;
        }
        return nums[lo];
    }
};
```

### 寻找两个正序数组的中位数

```C++
class Solution {
    int getKth(vector<int>& nums1, vector<int>& nums2, int k) {
        int m = nums1.size(), n = nums2.size();
        int idx1 = 0, idx2 = 0;
        while (true) {
            // 边界
            if (idx1 == m)
                return nums2[idx2 + k - 1];
            if (idx2 == n)
                return nums1[idx1 + k - 1];
            if (k == 1)
                return min(nums1[idx1], nums2[idx2]);

            // 新下标，每次右滑k/2-1，不能越界
            int _idx1 = min(idx1 + k / 2 - 1, m - 1);
            int _idx2 = min(idx2 + k / 2 - 1, n - 1);
            // 淘汰idx-_idx+1个元素
            if (nums1[_idx1] <= nums2[_idx2]) {
                k -= _idx1 - idx1 + 1;
                idx1 = _idx1 + 1;
            } else {
                k -= _idx2 - idx2 + 1;
                idx2 = _idx2 + 1;
            }
        }
    }

public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int len = nums1.size() + nums2.size();
        if (len % 2)
            return getKth(nums1, nums2, len / 2 + 1);
        else
            return (getKth(nums1,nums2,len/2)+getKth(nums1,nums2,len/2+1)) / 2.0;
    }
};
```

## 栈

### 有效的括号

```C++
class Solution {
public:
    bool isValid(string s) {
        if (s.size() % 2)
            return false;
        stack<int> stk;
        for (const char& x : s) {
            switch (x) {
            case '(':
            case '[':
            case '{':
                stk.emplace(x);
                continue;
            }

            if (stk.empty())
                return false;
            char t = stk.top();
            if (t == '(' && x == ')' || t == '[' && x == ']' ||
                t == '{' && x == '}')
                stk.pop();
            else
                return false;
        }
        return stk.empty() ? true : false;
    }
};
```

### 最小栈

```C++
class MinStack {
    stack<int> stk;
    stack<int> min_stk;

public:
    MinStack() { min_stk.emplace(INT_MAX); }

    void push(int val) {
        stk.emplace(val);
        min_stk.emplace(min(min_stk.top(), val));
    }

    void pop() {
        stk.pop();
        min_stk.pop();
    }

    int top() { return stk.top(); }

    int getMin() { return min_stk.top(); }
};
```

### 字符串解码

```C++
class Solution {
    // 读取数字
    string getDigit(const string& s, size_t& ptr) {
        string str;
        while (isdigit(s[ptr]))
            str.push_back(s[ptr++]);
        return str;
    }
    // 提取并合并为单个字符串
    string combineString(vector<string>& s) {
        string str;
        for (const auto& x : s)
            str += x;
        return str;
    }

public:
    string decodeString(string s) {
        vector<string> stk;
        size_t ptr;
        while (ptr < s.size()) {
            char ch = s[ptr];
            // 数字  字母    左括号
            if (isdigit(ch)) {
                string str = getDigit(s, ptr);
                stk.emplace_back(str);
            } else if (isalpha(ch) || ch == '[') {
                stk.emplace_back(string(1, ch));
                ++ptr;
            } else {
                ++ptr;
                vector<string> temp;
                while (stk.back() != "[") {
                    temp.emplace_back(stk.back());
                    stk.pop_back();
                }
                // 弹出'['
                stk.pop_back();
                // 反转
                reverse(temp.begin(), temp.end());
                // 提取
                string str1, str2 = combineString(temp);
                // 弹出数字
                int f = stoi(stk.back());
                stk.pop_back();
                // 构造串
                while (f--)
                    str1 += str2;

                stk.emplace_back(str1);
            }
        }
        return combineString(stk);
    }
};
```

### 每日温度

```C++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> predict(n);
        stack<int> stk;
        for (int i = 0; i < n; ++i) {
            // 温度严格递增则入栈
            while (!stk.empty() && temperatures[i] > temperatures[stk.top()]) {
                int preId = stk.top();
                predict[preId] = i - preId;
                stk.pop();
            }
            stk.emplace(i);
        }
        return predict;
    }
};
```

### 柱状图中最大的矩形

```C++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        // 添加边界
        heights.insert(heights.begin(), 0);
        heights.emplace_back(0);

        int n = heights.size(), maxS = 0;
        // 单调递增栈
        stack<int> stk;
        // 枚举高度
        for (int i = 0; i < n; ++i) {
            // 计算山顶宽度
            while (!stk.empty() && heights[stk.top()] > heights[i]) {
                int cur = stk.top();
                stk.pop();
                //(左边界,右边界)
                int L = stk.top() + 1;
                int R = i - 1;
                maxS = max(maxS, (R - L + 1) * heights[cur]);
            }
            stk.emplace(i);
        }
        return maxS;
    }
};
```

## 堆

### 数组中的第K个最大元素 

```C++
class Solution {
    void adjust(vector<int>& nums, int i, int n) {
        int L = 2 * i + 1, R = 2 * i + 2, k = i;
        if (L < n && nums[L] > nums[k])     k = L;
        if (R < n && nums[R] > nums[k])     k = R;
        if (k != i) {
            swap(nums[i], nums[k]);
            adjust(nums, k, n);
        }
    }
    void buildHeap(vector<int>& nums, int n) {
        for (int i = n / 2 - 1; i >= 0; --i)
            adjust(nums, i, n);
    }

public:
    int findKthLargest(vector<int>& nums, int k) {
        int n = nums.size();
        buildHeap(nums, n);

        for (int i = n - 1; i >= nums.size() - k + 1; --i) {
            swap(nums[i], nums[0]);
            adjust(nums, 0, --n);
        }
        return nums[0];
    }
};
```

### 前K个高频元素

```C++
class Solution {
    // 比较器
    static bool cmp(pair<int, int>& p, pair<int, int>& q) {
        return p.second > q.second;
    }

public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 用哈希映射记录频度
        unordered_map<int, int> mp;
        for (const int& x : nums)
            ++mp[x];
        // 优先队列
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(&cmp)>
            q(cmp);

        for (const auto& [x, count] : mp) {
            // 堆满，比较堆顶
            if (q.size() == k) {
                if (q.top().second < count) {
                    q.pop();
                    q.emplace(x, count);
                }
            } else {
                q.emplace(x, count);
            }
        }

        vector<int> v;
        while (!q.empty()) {
            v.emplace_back(q.top().first);
            q.pop();
        }
        return v;
    }
};
```

### 数据流中的中位数

```C++
class MedianFinder {
    priority_queue<int, vector<int>, less<int>> queL; // 中位数左区间
    priority_queue<int, vector<int>, greater<int>> queR;    // 中位数右区间
public:
    MedianFinder() {}

    void addNum(int num) {
        if (queL.empty() || num <= queL.top()) {
            queL.emplace(num);
            if (queR.size() + 1 < queL.size()) {
                queR.emplace(queL.top());
                queL.pop();
            }
        } else {
            queR.emplace(num);
            if (queR.size() > queL.size()) {
                queL.emplace(queR.top());
                queR.pop();
            }
        }
    }

    double findMedian() {
        if (queL.size() > queR.size())
            return queL.top();
        return (queL.top() + queR.top()) / 2.0;
    }
};
```

## 贪心

### 买卖股票

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int low = prices[0], profits = 0;
        for (const int& x : prices) {
            profits = max(profits, x - low);
            low = min(low, x);
        }
        return profits;
    }
};
```

### 跳跃游戏

```C++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = nums.size(), ed = 0;
        for (int i = 0; i < n; ++i) {
            if (i <= ed) {
                ed = max(ed, i + nums[i]);
                if (i == n - 1)     return true;
            }
        }
        return false;
    }
};
```

### 跳跃游戏II

```C++
class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size(), far = 0, ed = 0, step = 0;
        for (int i = 0; i < n - 1; ++i) {
            far = max(far, i + nums[i]);
            if (i == ed) {
                ++step;
                ed = far;
            }
        }
        return step;
    }
};
```

### 划分字母区间

```C++
class Solution {
public:
    vector<int> partitionLabels(string s) {
        int last[26];
        int n = s.size(), bg = 0, ed = 0;
        // 记录每个字符出现的最后位置
        for (int i = 0; i < n; ++i)
            last[s[i] - 'a'] = i;

        vector<int> v;
        for (int i = 0; i < n; ++i) {
            ed = max(ed, last[s[i] - 'a']);
            if (i == ed) {
                v.emplace_back(ed - bg + 1);
                bg = ed + 1;
            }
        }
        return v;
    }
};
```







## 其他

### 输入流提取数据

```C++
#include <iostream>		//标准输入输出流
#include <sstream>		//字符串流
#include <iomanip>		//std::hex转16进制
#include <string>
#include <vector>

vector<int> read_() {
    //输入：单行整形且以空格分割
    vector<int> nums;
    int a;
    while(cin >> a)
        nums.emplace_back(a);
    //输入：两行，且第一行声明第二行元素个数
    int n;
    cin >> n;
    vector<int> nums(n);
    for(int i=0;i<n;++i)
        cin >> nums[i];
    //输入：多行整形且以空格分割
    string line;
    vector<vector<int>> matrix;
    while(getline(cin,line))
    {
        istringstream iss(line);
        int a;
        vector<int> v;
        while(iss >> a)
            v.emplace_back(a);
        matrix.emplace_back(v);
    }
    //输入：单行点分十进制
	string line, hexIP = "";
	getline(cin, line);
	istringstream iss(line);
	ostringstream oss;
	int a;
	char ch;
	while (iss >> a)
	{
		oss << setw(2) << setfill('0') << hex << uppercase << a;
		iss >> ch;
	}
	hexIP = oss.str();
}
```

### 只出现一次的数字

```C++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int elem = 0;
        for (const int& x : nums)
            elem ^= x;
        return elem;
    }
};
```

### 多数元素

```C++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int votes = 0, x = 0;
        for (const int& t : nums) {
            // 正负抵消，下一个默认众数
            if (votes == 0)     x = t;
            votes += t == x ? 1 : -1;
        }
        return x;
    }
};
```

### 颜色分类

```C++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int p0 = 0, p2 = nums.size() - 1;
        // 双指针
        for (int i = 0; i <= p2; ++i) {
            // 保证当前元素扫描到位
            while (i <= p2 && nums[i] == 2)
                swap(nums[i], nums[p2--]);
            if (nums[i] == 0)
                swap(nums[i], nums[p0++]);
        }
    }
};
```

### 下一个排列

```C++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i = nums.size() - 2;
        // 找到首个升序邻居[i,i+1]
        while (i >= 0 && nums[i] >= nums[i + 1])
            --i;
        // 找到首个大于i且最靠右的j，交换
        if (i >= 0) {
            int j = nums.size() - 1;
            while (nums[j] <= nums[i])      --j;
            swap(nums[i], nums[j]);
        }
        // 逆置[i+1,n)
        reverse(nums.begin() + i + 1, nums.end());
    }
};
```

### 寻找重复数

```C++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int p = 0, q = 0;
        // 对数组建图，龟兔赛跑算法，索引跳转次数决定快慢
        do {
            p = nums[p];
            q = nums[nums[q]];
        } while (p != q);
        p = 0;
        while (p != q) {
            p = nums[p];
            q = nums[q];
        }
        return p;
    }
};
```

