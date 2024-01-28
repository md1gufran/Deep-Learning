#include <iostream>
#include <set>
#include <list>
#include <unordered_map>
#include <queue>
using namespace std;

void adjprep(unordered_map<int,list<int>>& adj,vector<pair<int,int>>& edges){
    for(auto i: edges){
        int u = edges[i].first;
        int v = edges[i].second;

        adj[u].push_back(v);
        adj[v].push_back(u);
    }
}


void bfs(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,vector<int>& ans,int node){
    queue<int> q;
    q.push(node);
    visited[node] = 1;
    while(!q.empty()){
        int front = q.front();
        q.pop();

        ans.push_back(front);
        for(auto i: adj[node]){
            if(!visited[i]){
                q.push(i);
                visited[i] = 1;
            }
        }
    }
}

//Keep in mind that this is for nodes numbered from 0 to n-1
int main(){
    int n;  //number of nodes
    cout << "Please enter number of nodes" << endl;
    cin >>  n;

    int m;
    cout << "Please enter number of edges" << endl;
    cin >> m;
    vector<pair<int,int>> edges;
    vector<int> ans;
    for(int i=0;i<m;i++){
        int u,v;
        cin >> u >> v;
        edges.push_back(make_pair(u,v));
    }  

    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited; 

    //we need this loop only when we have a graph which might be disconnected like having different component
    for(int i=0;i<n;i++){
        if(!visited[i]){
            bfs(adj,visited,ans,i);
        }
    }

    return ans;
}
