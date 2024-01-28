#include <iostream>
#include <list>
#include <unordered_map>
#include <queue>
#include <stack>
using namespace std;

int main()
{
    int e,v;
    cin >> e >> v;
    unordered_map<int,list<int>> adj;
    vector<int> indegree(v,0);
    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
        indegree[x]++;
    }

    queue<int> q;
    for(int i=0;i<v;i++){
        if(indegree[i]==0) q.push(i);
    }
    vector<int> ans;
    int cnt = 0;
    while (!q.empty())
    {
        int front = q.front();
        q.pop();
        ans.push_back(front);
        for(auto i:adj[front]){
            indegree[i]--;
            if(indegree[i]==0) q.push(i);
        }
    }
    
    for(auto i: ans) cout << i << " ";
    //as we know that topsort is just for DAG
    // to detect cyle using this we can say that if cnt==node than acyclic
    //but if cnt!=node than we can say that it cyclic graph
    if(cnt==v) cout << "acyclic graph";
    else cout << "Cyclic graph";
}