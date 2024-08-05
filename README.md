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

