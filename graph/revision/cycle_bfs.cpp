#include <iostream>
#include <list>
#include <unordered_map>
#include <queue>
using namespace std;

bool cycle(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,unordered_map<int,int>& parent,int node)
{
    queue<int> q;
    q.push(node);
    visited[node] = true;
    parent[node] = -1;
    while (!q.empty())
    {
        int front = q.front();
        q.pop();
        for(auto i:adj[front]){
            if(visited[i]==true && parent[front]!=i){
                return true;
            }else if(!visited[i]){
                q.push(i);
                visited[i] = true;
                parent[i] = front;
            }
        }
    }
    return false;
}

bool dfscycle(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,unordered_map<int,int>& parent,int node)
{
    visited[node] = true;
    
    for(auto i:adj[node]){
        if(visited[i]==true && parent[node]!=i){
            return true;
        }else if(!visited[i]){
            parent[i] = node;
            dfscycle(adj,visited,parent,i);
        }
    }
    return false;
}
int main()
{
    int e,v;
    cin >> e >> v;

    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;
    unordered_map<int,int> parent;

    //for undirected graph
    for(int i=0;i<e;i++){
        int u,x;
        cin >> u >> x;
        adj[u].push_back(x);
        adj[u].push_back(x);
    }
}