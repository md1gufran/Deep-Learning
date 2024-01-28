#include <iostream>
#include <unordered_map>
#include <list>
#include <queue>
using namespace std;

bool iscyclic(unordered_map<int,list<int>>& adj,unordered_map<int,bool>& visited,int src)
{
    unordered_map<int,int> parent;
    parent[src] = -1;
    visited[src] = 1;
    queue<int> q;
    q.push(src);

    while (!q.empty())
    {
        int front = q.front();
        q.pop();

        for(auto i:adj[front]){
            if(visited[i]==1 && parent[front]!=i){
                return true;
            }else{
                q.push(i);
                visited[i] = 1;
                parent[i] = front;
            }
        }
    }
    return false;
}

int main(){
    int n;  // number of edges
    cin >> n;

    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;

    for(int i=0;i<n;i++){
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    bool ans;
    for(int i=0;i<n;i++){
        if(!visited[i]){
            ans = iscyclic(adj,visited,i);
            if(ans) break;
        }
    }

    if(ans) cout << "Yes"<<endl;
    else cout << "NO" << endl;
}