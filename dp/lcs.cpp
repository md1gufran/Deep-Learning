#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int lcs(int i,int j,string str,string str1,vector<vector<int>>& dp,string& ans)
{
    if(str[i]=='\0' || str[j]=='\0') return 0;
    if(dp[i][j]!=-1){
        return dp[i][j];
    }
    if(str[i]==str1[j]){
        ans.push_back(str[i]);
        return dp[i][j] = 1 + lcs(i+1,j+1,str,str1,dp,ans);
    }
    
    return dp[i][j] = max(lcs(i+1,j,str,str1,dp,ans),lcs(i,j+1,str,str1,dp,ans));
    
}
int main()
{
    string str,str1;
    cin >> str >> str1; //our objective is to find longest common subsequence
    string ans;
    //first apporach dynamic programming
    vector<vector<int>> dp(str.size()+1, vector<int>(str1.size()+1, -1));

    cout << lcs(0,0,str,str1,dp,ans);
    cout << endl;
    reverse(ans.begin(),ans.end());
    cout << ans << endl;
    
    //we can also use two for loops for doing the same task
    return 0;
}