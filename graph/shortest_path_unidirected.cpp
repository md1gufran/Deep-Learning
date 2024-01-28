//shortes path in undirected graph assuming it is unweighted graph
//bfs traveral first we will find the parents of each node and then back track with that parent 
// from target to soruce and then reverse the ans array
#include <iostream>
#include <unordered_map>
#include <list>
#include <queue>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    //edges,vertex,source,end node
    int e,v,s,t;

    cout << "Enter number of edges" << endl;
    cin >> e;
    cout << "Enter number of vertex" << endl;
    cin >> v;
    cout << "Starting point" << endl;
    cin >> s;
    cout << "Please target point" << endl;
    cin >> t;

    cout << endl;
    unordered_map<int,list<int>> adj;
    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
        adj[x].push_back(u);
    }
    unordered_map<int,bool> visited;
    unordered_map<int,int> parent;
    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;
    while(!q.empty()){
        int front = q.front();
        q.pop();

        for(auto i:adj[front]){
            if(!visited[i]){
                visited[i] = true;
                parent[i] = front;
                q.push(i);
            }
        }
    }
    //now we will backtrack to get the shortest path
    vector<int> ans;
    int curr = t;
    ans.push_back(curr);
    while(curr!=s){
        curr = parent[curr];
        ans.push_back(curr);
    }

    reverse(ans.begin(),ans.end());
    for(auto i: ans) cout << i << " ";
    cout << endl;
    return 0;
}