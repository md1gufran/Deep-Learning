//bfs traversal for kahn algo for topological sort
#include <iostream>
#include <list>
#include <queue>
#include <vector>
#include <unordered_map>
using namespace std;

int main()
{
    int e,v;  //number of edges
    cout << "Enter the number of edges" << endl;
    cin >> e;
    cout << "Enter your vertex" << endl;
    cin >> v;
    unordered_map<int,list<int>> adj;

    //directed graph
    for(int i=0;i<e;i++){
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    vector<int> indegree(v);
    for(auto i: adj){
        for(auto j:i.second){
            indegree[j]++;
        }
    }

    //bfs traversal using kahn algo
    //0 degree walo ko push kardo
    queue<int> q;
    for(int i=0;i<v;i++){
        if(indegree[i]==0)
            q.push(i);
    }

    vector<int> ans;
    while (!q.empty())
    {
        int front = q.front();
        q.pop();
        ans.push_back(front);

        //update karo indegree of neighbour and check whether it is 0 or not    
        for(auto i:adj[front]){
            indegree[i]--;
            if(indegree[i]==0)
                q.push(i);
        }
    }
    
    for(auto i: ans) cout << i << " ";
    cout << endl;
    return 0;
}