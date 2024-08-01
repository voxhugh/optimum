<center>
    	<h1>笔试框架</h1>
</center>


## 岛屿(DFS)

```C++
class Solution{
	void dfs(vector<vector<int>>& nums,int i,int j)
    {
        //base case
        if(!inside(nums,i,j))	return;
        if(nums[i][j]!=1)	return;
        nums[i][j] = 2;
        
	dfs(nums,i-1,j);
        dfs(nums,i+1,j);
        dfs(nums,i,j-1);
        dfs(nums,i,j+1);
	}
    bool inside(vector<vector<int>>& nums,int i,int j)
    {
        return i>=0&&i<nums.size()&&j>=0&&j<nums[0].size();
	}
public:
    int isLands(vector<vector<int>>& nums)
    {
        int m = nums.size();
        if(!m)	return 0;
        int n = nums[0].size();
        
        for(int i=0;i<m;++i)
            for(int j=0;j<n;++j)
                if(nums[i][j]==1)	dfs(nums,i,j);
    }
};
```

## 烂橘子(BFS)

```C++
class Solution{
    int org;
    int time[10][10];
    int x[4] = {-1, 1, 0, 0};
    int y[4] = {0, 0, -1, 1};
public:
    int orangeRotting(vector<vector<int>>& nums)
    {
        queue<pair<int,int>> Q;
        memset(time,-1,sizeof(time));
        org = 0;
        int m = nums.size(),n = nums[0].size(),ans = 0;
        //存入烂橘子
        for(int i=0;i<m;++i)
            for(int j=0;j<n;++j)
            {
                if(nums[i][j]==2)
                {
                    Q.emplace(i,j);
                    time[i][j] = 0;
		}
                else if(nums[i][j]==1)
                    ++org;
	    }
        //腐烂
        while(!Q.empty())
        {
            auto [r,c] = Q.front();
            Q.pop();
            //上 下 左 右
            for(int i=0;i<4;++i)
            {
                int tx = r+x[i];
                int ty = c+y[i];
                //越界，访问过，空格
                if(tx<0||tx>=m||ty<0||ty>=n||time[tx][ty]!=-1||!time[tx][ty])
                    continue;
                //时间推进
                time[tx][ty] = time[r][c]+1;
                //入队
                Q.emplace(tx,ty);
                if(nums[tx][ty]==1)
                {
                    --org;
                    ans = time[tx][ty];
                    if(!org)	break;
		}
	      }
	}
        return org?-1:ans;    
	}
};
```

## 异位词(hash)

```C++
class Solution{
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs)
    {
        unordered_map<string,vector<string>> mp;
        for(string& x:strs)
        {
            string key = x;
            sort(key.begin(),key.end());
            mp[key].emplace_back(x);
		}
        vector<vector<string>> ans;
        for(auto it = mp.begin();it!=mp.end();++it)
            ans.emplace_back(it->second);
        return ans;
    }
};
```

## 盛水(pp)

```C++
class Solution{
public:
    int maxArea(vector<int>& height)
    {
        int i = 0, j = height.size()-1, ans = 0;
        while(i<j)
        {
            ans = height[i] < height[j]?
                max(ans,(j-i)*height[i++]):
            	max(ans,(j-i)*height[j--])
	}
        return ans;
     }
};
```

## 和为K子数组(subString)

```C++
class Solution{
public:
    int subarraySum(vector<int>& nums,int k)
    {
        unordered_map<int,int> mp;
        mp[0] = 1;
        int count = 0, pre = 0;
        for(int& x:nums)
        {
            pre += x;
            if(mp.count(pre-k))
                count += mp[pre-k];
            ++mp[pre];
	}
        return count;
    }
};
```

## 自身外数组积(array)

```C++
class Solution{
  public:
    vector<int> productExceptSelf(vector<int>& nums)
    {
        int n = nums.size();
        vector<int> ans(n);
        
        ans[0] = 1;
        for(int i=1;i<n;++i)
      		ans[i] = nums[i-1]*ans[i-1];
        
        int R = 1;
        for(int i=n-1;i>=0;--i)
        {
            ans[i] = ans[i]*R;
            R *= nums[i];
        }
        return ans;
    }
};
```

## 旋转图像(matrix)

```C++
class Solution{
public:
    void rotate(vector<vector<int>>& matrix)
    {
        int n = matrix.size();
        
        for(int i=0;i<n/2;++i)
            for(int j=0;j<n;++j)
                swap(matrix[i][j],matrix[n-i-1][j]);
        
        for(int i=0;i<n;++i)
            for(int j=0;j<i;++j)
                swap(matrix[i][j],matrix[j][i]);
    }
}
```

## 矩阵搜索(Z)

```C++
class Solution{
public:
    bool searchMatrix(vector<vector<int>>& matrix,int k)
    {
        int m = matrix.size(), n = matrix[0].size();
        int x = 0, y = n-1;
        while(x<m && y>=0)
        {
            if(matrix[x][y]==k)	return true;
            if(matrix[x][y]>k)	--y;
            else	++x;
        }
        return false;
    }
}
```

## 第k大元素(堆排)

```C++
class Solution{
	void adjust(vector<int>& a,int i,int heapSize)
    {
        int L = i*2+1,R = i*2+2,lg = i;
        if(L<heapSize && a[L]>a[lg])	lg = L;
        if(R<heapSize && a[R]>a[lg])	lg = r;
        if(lg!=i)
        {
            swap(a[i],a[lg]);
            adjust(a,lg,heapSize);
        }
    }
    void crtMaxHeap(vector<int>& a,int heapSize)
    {
        for(int i=heapSize/2;i>=0;--i)
            adjust(a,i,heapSize);
    }
public:
    int findKthLargest(vector<int>& nums,int k)
    {
        int heapSize = nums.size();
        crtMaxHeap(nums,heapSize);
        for(int i=heapSize-1;i >= nums.size()-k+1;--i)
        {
            swap(nums[i],nums[0]);
            adjust(nums,0,--heapSize);
        }
        return nums[0];
    }
};
```

## 跳一跳(greed)

```C++
class Solution{
public:
    int jump(vector<int>& nums)
    {
        int maxPos = 0, n = nums.size(), end = 0, step = 0;
        for(int i=0;i<n-1;++i)
        {
            if(maxPos >= i)
            {
                maxPos = max(maxPos, i+nums[i]);
                if(i == end)
                {
                    end = maxPos;
                    ++step;
                }
            }
	}
        return step;
    }
};
```

## 小偷(DP)

```C++
class Solution{
public:
    int rob(vector<int>& nums)
    {
        if(nums.empty())	return 0;
        int n = nums.size();
        if(n == 1)	return nums[0];
        
        vector<int> dp(n);
        dp[0] = nums[0];
        dp[1] = max(nums[0],nums[1]);
        for(int i=2;i<n;++i)
            dp[i] = max(dp[i-2] + nums[i], dp[i-1]);
        return dp[size-1];
    }
};
```
## 单词搜索(BT)

```C++
class Solution{
	int rows,cols;
    bool dfs(vector<vector<char>>& board,string word,int i,int j,int k)
    {
        if(i>=rows||i<0||j>=cols||j<0||board[i][j]!=word[k])
            return false;
        if(k==word.size()-1)	return true;
        board[i][j] = '\0';
        bool ans = dfs(borad,word,i+1,j,k+1)||
            		dfs(borad,word,i-1,j,k+1)||
            		dfs(borad,word,i,j+1,k+1)||
            		dfs(board,word,i,j-1,k+1);
        board[i][j] = word[k];
        return ans;  
    }
public:
    bool exist(vector<vector<char>>& board,string word)
    {
        rows = board.size();
        cols = board[0].size();
        for(int i=0;i<rows;++i)
            	for(int j=0;j<cols;++j)
                    if(dfs(board,word,i,j,0))
                        return true;
        return false;
    }
};
```

## 旋转数组(BinSearch)

```C++
class Solution{
public:
    int search(vector<int>& nums,int k)
    {
        int lo = 0,hi = nums.size()-1;
        while(lo<hi)
        {
            int mid = (lo+hi)/2;
            if((nums[0]>k)^(nums[0]>nums[mid])^(k>nums[mid]))
                lo = mid+1;
            else
                hi = mid;
        }
        return lo ==hi && nums[lo]== k?lo:-1;
    }
}
```
