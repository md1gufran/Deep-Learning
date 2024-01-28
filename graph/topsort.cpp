//This is topological sort using dfs approach
//Keep in mind that dfs call is in the order of topological sort
//This is only valid for DAG - > Directed acyclic graph
#include <iostream>
#include <unordered_map>
#include <list>
#include <queue>
#include <stack>
using namespace std;

void topsort(unordered_map<int,bool>& visited,unordered_map<int,list<int>>& adj,int node,stack<int>& st)
{
    visited[node] =true;

    for(auto neighbour:adj[node]){
        if(!visited[neighbour]){
            topsort(visited,adj,neighbour,st);
        }
    }
    st.push(node);
}
int main()
{
    int n;  //number of edges
    cin >> n;

    unordered_map<int,list<int>> adj;
    unordered_map<int,bool> visited;
    stack<int> st;

    //undirected graph
    for(int i=0;i<n;i++){
        int u,v;
        cin >> u >>v;
        adj[u].push_back(v);
    }

    for(int i=0;i<n;i++){
        if(!visited[i]){
            topsort(visited,adj,i,st);
        }
    }

    while (!st.empty())
    {
        cout << st.top() << " ";
        st.pop();
    }
    
    cout << endl;
    return 0;
}