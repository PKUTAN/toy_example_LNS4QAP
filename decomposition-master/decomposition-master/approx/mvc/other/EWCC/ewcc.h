/************************************************
** Date:   2010.6.4  
** EWLS (Edge Weight Local Search) 
** Author: Shaowei Cai, shaowei_cai@126.com    
**		   School of EECS, Peking University   
**		   Beijing, China                      
**                                                         
************************************************/

/************************************************
** Date:	2010.9.8 
** EWCC (Edge Weight Congiguration Checking)
** Author: Shaowei Cai, shaowei_cai@126.com  
** Detail:	Improve EWLS to EWCC by utilizing the configuration checking strategy to handle 
**			the cycling problem in local search. 
**			On selecting a vertex to add into C, for a vertex v not in C, if the configuration 
**			of the induced subgraph G[N [v]] never changes (which means the circumstance of v 
**			never changes) since v’s last removing from C, it should not be added back to C.
************************************************/

/** ------------------------------------------- **/
/** Consult README for usage and input format   **/


#include <iostream>
#include <fstream>
#include <cstdlib>
#include "mersenne.cc"
#include <sys/times.h>
#include <cmath>

using namespace std;

Mersenne rng;
tms start, finish;

/******************** Notations and definitios *************************
Input G= (V, E),V = {1,2,...,n}. For e(v1, v2), endpoint(e) = {v1, v2).

During the search procedure, EWLS always maintains a current 
candidate solution C and a set L of edges not covered by C.

For a candidate solution P, cost(G, P) indicates the total weights 
of edges not covered by P.

dscore(v) = cost(G,C) - cost(G,C')
where C' = C\{v} if v in C, and C' = C \/ {v} otherwise.

For two vertices u, v in V , where u in C and v not in C,
score(u, v) = cost(G,C) ? cost(G, [C\{u}]\/{v})
which indicates the score of exchanging u and v.

Lemma Given G = (V,E) and C is the current candidate
solution, for a pair of vertices u, v in V , where u in C and
v not in C, score(u, v) = dscore(u) + dscore(v)+w(e(u, v))
if e(u, v) in E; and score(u, v) = dscore(u) + dscore(v) else.

**********************************************************************/

struct Edge{
	int v1;
	int v2;
};

struct Number{
	int num;
	Number* pre;
	Number* next;
};

/*max vertex count and max edge count*/
#define	MAXV	10000
#define MAXE	20000000

/*parameters of algorithm*/
long long		max_steps;				//step limit
int		optimal_size;			//terminate the algorithm before step limit if it finds a vertex cover of optimal_size

/*parameters of the instance*/
int		v_num;//|V|
int		e_num;//|E|

/*structures about edge*/
Edge	edge[MAXE];
int		edge_weight[MAXE];
int		cover_vertex_count_of_edge[MAXE];//amount of endpoints in C of an edge e

/*structures about vertex*/
int		dscore[MAXV];			//dscore of v
int		highest_dscore_c;		//the highest dscore in C (that is the improvement with minimum absolute value)
int		v_highest_dscore_c;		//the vertex of the highest dscore in C

//from vertex to it's edges
int		v_edges[MAXV][MAXV];	//edges related to v, v_edges[i][k] means vertex v_i's k_th edge
int		edge_num_of_v[MAXV];	//amount of edges related to v

//from vertex to its adjacent vertices
int		v_adj[MAXV][MAXV];		//v_adj[v_i][k] = v_j(actually, that is v_i's k_th neighbor)

//from two vertex to the edge between them
int		edge_between_vertices[MAXV][MAXV];

/* structures about solution */
int		c_size;						//cardinality of C
int		v_in_c[MAXV];				//a flag indicates whether a vertex is in C
int		best_v_in_c[MAXV];			//a flag indicates whether a vertex is in best solution
long long		step;

//uncovered edge list
Number*	l_head;						//L
Number*	edge_ptr[MAXE];				//array of edge pointer (static), edge_ptr[i] always refer to edge i 
int		l_size;						//|L|

//taboo
int		tabu_remove;

/* functions declaration */
void build_instance(ifstream& infile);
void read_edge(ifstream& in);
void init_sol();
void cover_LS();
int  choose_exchange_pair(int v1, int v2, int& add_v, int& remove_v);
int  choose_remove_vertex(int v0, int& remove_v);//v0==v1 || v0==v2
void add(int v);
void remove(int v);
void update_edge_weight();
void cover_rest_edges();
int check_solution();


void print_solution()
{
	int cover_vertex_count=0;

	for (int i=1; i<=v_num; i++)
	{
		if (best_v_in_c[i]==1)
			cover_vertex_count++;
		else if (best_v_in_c[i]!=1)//output max independent set
			cout<<i<<'\t';
	}
	cout<<endl;
}


void update_best_sol()
{
	int i;

	for (i=1;i<=v_num;i++)
		best_v_in_c[i] = v_in_c[i];
	
	//output time to get this optimal solution
	times(&finish);
	cout<<"optimal mvc: "<<c_size<<'\t';
	double comp_time = double(finish.tms_utime - start.tms_utime + finish.tms_stime - start.tms_stime)/sysconf(_SC_CLK_TCK);
	comp_time = round(comp_time * 100)/100.0;
	cout<<"in "<<step<<" steps, "<<comp_time<<" seconds"<<endl;

}


void update_best_improvement_of_coverset()
{
	highest_dscore_c = -100000000;
	for (int v=1; v<=v_num; ++v )
	{
		if (dscore[v]>highest_dscore_c && v_in_c[v]==1)
		{
			highest_dscore_c = dscore[v];
			v_highest_dscore_c = v;
		}
	}

}


void build_instance(ifstream& infile)
{
	char line[1024];
	char tempstr1[10];
	char tempstr2[10];
	int  v;

	/*** build problem data structures of the instance ***/
	infile.getline(line,1024);
	while (line[0] != 'p')
		infile.getline(line,1024);
	sscanf(line, "%s %s %d %d", tempstr1, tempstr2, &v_num, &e_num);


	//initiate problem data structures
	for (v=1; v<=v_num; v++)
	{
		edge_num_of_v[v] = 0;

		for (int u=1; u<=v_num; u++)
			edge_between_vertices[v][u] = e_num;//init, edge[e_num] means no exits such an edge
	}
	edge_weight[e_num] = 0;
	edge[e_num].v1 = edge[e_num].v2 = -1; 

	//read edge to build problem data structures
	read_edge(infile);
	infile.close();

	Number* ptr_edgeNumber;
	int e;
	for (e=e_num; e>=0; e--)//e_num is an end bound, the real edges index from 0 to e_num-1
	{
		ptr_edgeNumber = new Number;
		ptr_edgeNumber->num = e;

		edge_ptr[e] = ptr_edgeNumber;
	}

}

void destroy_instance()
{
	int e;
	for (e=0; e<=e_num; e++)//e_num is an end bound, the real edges index from 0 to e_num-1
	{
		if(edge_ptr[e]!=NULL)
			delete edge_ptr[e];
	}
}

void read_edge(ifstream& fin)
{
	int		e;
	char	tmp;
	int		v1,v2;

	for (e=0; e<e_num; e++)
	{
		fin>>tmp;
		fin>>v1>>v2;

		edge[e].v1 = v1;
		edge[e].v2 = v2;

		v_edges[v1][edge_num_of_v[v1]] = e;
		v_edges[v2][edge_num_of_v[v2]] = e;

		v_adj[v1][edge_num_of_v[v1]] = v2;
		v_adj[v2][edge_num_of_v[v2]] = v1;

		edge_num_of_v[v1]++;
		edge_num_of_v[v2]++;

		edge_between_vertices[v1][v2] = edge_between_vertices[v2][v1] = e;
	}
}



void update_target_size()
{
	c_size--;
	
	int max_improvement;
	int max_vertex;//vertex with the highest improvement in C

	max_improvement=-100000000;
	for (int v=1; v<=v_num; ++v)
	{
		if(v_in_c[v]==0)continue;
		if (dscore[v]>max_improvement)
		{
			max_improvement = dscore[v];
			max_vertex = v;
		}
	}
	remove(max_vertex);
	
	
	while(l_size==0)
	{
		update_best_sol();
		
		if (c_size==optimal_size)
			return;
		
		update_target_size();

	}
}

int conf_change[MAXV];

void init_sol()
{
	int i,v,e;

	/*** build solution data structures of the instance ***/
	//init vertex cover
	for (v=1; v<=v_num; v++)
	{
		v_in_c[v] = 0;
		dscore[v] = 0;
		
		conf_change[v] = 1;
	}

	for (e=0; e<e_num; e++)
	{
		edge_weight[e] = 1;
		dscore[edge[e].v1]+=edge_weight[e];
		dscore[edge[e].v2]+=edge_weight[e];
	}

	//init uncovered edge list
	Number* ptr_edgeNumber;
	l_head = NULL;
	for (e=e_num; e>=0; e--)//e_num is an end bound, the real edges index from 0 to e_num-1
	{
		ptr_edgeNumber = edge_ptr[e];

		ptr_edgeNumber->next = l_head;
		ptr_edgeNumber->pre = NULL;
		if (l_head!=NULL)
			l_head->pre = ptr_edgeNumber;
		l_head = ptr_edgeNumber;
	}
	l_size = e_num;

	for (e=0; e<e_num; e++)
		cover_vertex_count_of_edge[e] = 0;

	int best_vertex_improvement;
	int best_count;
	int best_array[MAXV];

	for (i=0; l_size>0; )
	{
		best_vertex_improvement = 0;
		best_count = 0;
		for (v=1; v<=v_num; ++v)
		{
			if(v_in_c[v]==1)continue;

			if (dscore[v]>best_vertex_improvement)
			{
				best_vertex_improvement = dscore[v];
				best_array[0] = v;
				best_count = 1;
			}
			else if (dscore[v]==best_vertex_improvement)
			{
				best_array[best_count] = v;
				best_count++;
			}
		}

		if(best_count>0)
		{
			add(best_array[rng.next(best_count)]);
			++i;
		}
	}

	c_size = i;
	
	update_best_sol();
	
	if(c_size==optimal_size)return;
	
	update_target_size();
	
	update_best_improvement_of_coverset();
	
}




void add(int v)
{
	v_in_c[v] = 1;
	dscore[v] = -dscore[v];
	
	if (dscore[v]>highest_dscore_c)
	{
		highest_dscore_c = dscore[v];
		v_highest_dscore_c = v;
	}

	int i,e;
	Number* ptr_edgeNum;
	int edges_count_of_vertex_v = edge_num_of_v[v];
	for (i=0; i<edges_count_of_vertex_v; ++i)
	{
		e = v_edges[v][i];
		++cover_vertex_count_of_edge[e];

		if (cover_vertex_count_of_edge[e]==1)//the other vertex of e isn't in cover set
		{
			dscore[v_adj[v][i]] -= edge_weight[e];
			conf_change[v_adj[v][i]] = 1;

			/*remove e out of uncovered_edge_list*/
			ptr_edgeNum = edge_ptr[e];
			if (ptr_edgeNum->pre!=NULL)//ptr_edgeNum isn't the head node
			{
				(ptr_edgeNum->pre)->next = ptr_edgeNum->next;
				(ptr_edgeNum->next)->pre = ptr_edgeNum->pre;
				ptr_edgeNum->pre = NULL;
			}
			else{
				l_head = ptr_edgeNum->next;
				l_head->pre = NULL;
			}
			--l_size;
		}
		else
		{
			dscore[v_adj[v][i]] += edge_weight[e];
			if (dscore[v_adj[v][i]]>highest_dscore_c)
			{
				highest_dscore_c = dscore[v_adj[v][i]];
				v_highest_dscore_c = v;//should be "v_highest_dscore_c = v_adj[v][i];" but it does not matter that much.
			}

		}
		
	}
}


int	need_update = 0;

void remove(int v)
{
	v_in_c[v] = 0;
	dscore[v] = -dscore[v];
	conf_change[v] = 0;

	if (v==v_highest_dscore_c)
	{
		//update_best_improvement_of_coverset();
		need_update++;
	}

	int i,e;
	Number* ptr_edgeNum;

	int edges_count_of_vertex_v = edge_num_of_v[v];
	for (i=0; i<edges_count_of_vertex_v; ++i)
	{
		e = v_edges[v][i];
		--cover_vertex_count_of_edge[e];

		if (cover_vertex_count_of_edge[e]==0)//the other vertex of e isn't in cover set
		{
			dscore[v_adj[v][i]] += edge_weight[e];
			conf_change[v_adj[v][i]] = 1;

			/*add e to uncovered_edge_list(add at the head)*/
			ptr_edgeNum = edge_ptr[e];
			ptr_edgeNum->next = l_head;
			l_head->pre = ptr_edgeNum;
			l_head = ptr_edgeNum;

			++l_size;
		}
		else
		{
			dscore[v_adj[v][i]] -= edge_weight[e];
			if (v_adj[v][i]==v_highest_dscore_c)
			{
				//update_best_improvement_of_coverset();
				need_update++;
			}
		}
	}
	
	if(need_update>0)
	{
		update_best_improvement_of_coverset();
		need_update = 0;
	}
}


//check whether the solution that EW-ILS found is a proper solution
int check_solution()
{
	int mis_solution[MAXV];
	int mis_v = 0;
	int v;
	for(v=1; v<=v_num; ++v)
	{
		if(best_v_in_c[v] != 1)//no in cover(means in mis)
		{
			mis_solution[mis_v] = v;
			mis_v++;
		}
	}
	for(int i=0; i<mis_v;i++)
	{
		for(int j=i+1; j<mis_v; j++)
		{
			if(edge_between_vertices[mis_solution[i]][mis_solution[j]] != e_num)//has an edge
			{
				return 0;
			}
		}
	}	
	return 1;
}

