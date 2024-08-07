<center>
    	<h1>笔试框架</h1>
</center>


[TOC]



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

## 普通数组

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

