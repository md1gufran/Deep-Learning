#include <iostream>
#include <list>
#include <unordered_map>
#include <vector>
using namespace std;

int main()
{
    int e,v1;
    cout << "Number of edges" << endl;
    cin >> e;
    cout << "Number of vertex" << endl;
    cin >> v1;

    //creaing directed graph
    unordered_map<int,list<int>> adj;

    for(int i=0;i<e;i++){
        int u,v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    //printing of directed graph
    for(auto i:adj){
        cout << i.first << " "; // printing the node and its respective children just for case not exactly
        for(auto j:i.second) cout << j << " ";
        cout << endl;
    }

    //unordered map for undirected graph
    unordered_map<int,list<int>> adj1;
    for(int i=0;i<e;i++){
        int u,v;
        cin >> u >> v;
        adj1[u].push_back(v);
        adj1[v].push_back(u);
    }

    //for printing we just use same for loop
    for(auto i:adj1){
        cout << i.first << " ";
        for(auto j:i.second){
            cout << j << " ";
        }
        cout << endl;
    }

    //now we will do by adjaceny matrix for undirected graph
    vector<vector<int>> v(v1,vector<int> (v1,0));

    for(int i=0;i<v1;i++){
        int u,x;
        cin >> u >> x;
        v[u][x] = 1;
        v[x][u] = 1;
    }
    //printing with the help of matrix
    for(int i=0;i<v1;i++){
        cout << i << " : " ;
        for(int j=0;i<v1;j++){
            if(v[i][j]==1) cout << j << " ";
        }
        cout << endl;
    }
    return 0;
}