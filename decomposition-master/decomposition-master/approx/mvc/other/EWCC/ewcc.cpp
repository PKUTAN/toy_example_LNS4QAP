
#include "ewcc.h"

//variables for choose_exchange_pair() function
int check_vertex_count;
int check_vertex[MAXV];//record the vertexes that been checked for exchange in after last add&remove operation(which change vertexes weight).

int add_array[MAXV];
int remove_array[MAXV];

int  choose_remove_vertex(int v0, int& remove_v)
{
	if(conf_change[v0]==0)return -1;
	
	int i;
	for (i=0; i<check_vertex_count;++i)
	{
		if(check_vertex[i]==v0)
			return -1;
	}

	int vertex_improvement_v0 = dscore[v0];
	if (vertex_improvement_v0+highest_dscore_c<=0)
	{	
		int exchange_count=0;
		int edges_count_of_vertex_v0 = edge_num_of_v[v0];
		for (i=0; i<edges_count_of_vertex_v0; ++i)
		{
			if (vertex_improvement_v0+dscore[v_adj[v0][i]]+edge_weight[v_edges[v0][i]]>0)
			{
				if (v_in_c[v_adj[v0][i]]==0||v_adj[v0][i]==tabu_remove)continue;
				remove_array[exchange_count] = v_adj[v0][i];
				exchange_count++;
			}
		}
		if(exchange_count>0)
		{
			remove_v = remove_array[rng.next(exchange_count)];
			return 1;
		}
		else
		{
			check_vertex[check_vertex_count] = v0;
			check_vertex_count++;
			return -1;
		}
	}
	else
	{
		//int v = rand()%v_num+1;
		int v = rng.next(v_num)+1;
		int init_v = v;

		//init_v to v_num
		for (; v<=v_num;++v)
		{
			if(vertex_improvement_v0+edge_weight[edge_between_vertices[v0][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				remove_v = v;
				return 1;
			}
		}
		//1 to init_v-1
		for (v=1; v<init_v;++v)
		{
			if(vertex_improvement_v0+edge_weight[edge_between_vertices[v0][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				remove_v = v;
				return 1;
			}
		}

		check_vertex[check_vertex_count] = v0;
		++check_vertex_count;

		return -1;
	}

}

int choose_exchange_pair_in_adj(int v1, int v2, int& add_v, int& remove_v)
{
	int i;
	int exchange_count=0;

	int vertex_improvement_v1 = dscore[v1];
	int vertex_improvement_v2 = dscore[v2];

	int edges_count_of_vertex_v1 = edge_num_of_v[v1];
	for (i=0; i<edges_count_of_vertex_v1; ++i)
	{
		if (vertex_improvement_v1+dscore[v_adj[v1][i]]+edge_weight[v_edges[v1][i]]>0)
		{
			if (v_in_c[v_adj[v1][i]]==0||v_adj[v1][i]==tabu_remove)continue;
			add_array[exchange_count] = v1;
			remove_array[exchange_count] = v_adj[v1][i];
			exchange_count++;
		}
	}
	if (exchange_count==0)
	{
		check_vertex[check_vertex_count] = v1;
		check_vertex_count++;
	}

	int edges_count_of_vertex_v2 = edge_num_of_v[v2];
	for (i=0; i<edges_count_of_vertex_v2; ++i)
	{
		if (vertex_improvement_v2+dscore[v_adj[v2][i]]+edge_weight[v_edges[v2][i]]>0)
		{
			if (v_in_c[v_adj[v2][i]]==0||v_adj[v2][i]==tabu_remove)continue;
			add_array[exchange_count] = v2;
			remove_array[exchange_count] = v_adj[v2][i];
			exchange_count++;
		}
	}

	if(exchange_count>0)
	{
		int r = rng.next(exchange_count);
		add_v = add_array[r];
		remove_v = remove_array[r];
		return 1;
	}
	else
	{
		check_vertex[check_vertex_count] = v2;
		check_vertex_count++;
		return -1;
	}
}


int choose_exchange_pair(int v1, int v2, int& add_v, int& remove_v)
{
	//check taboo
	if (conf_change[v1]==0)
	{
		add_v = v2;
		return choose_remove_vertex(v2,remove_v);
	}
	else if(conf_change[v2]==0)
	{
		add_v = v1;
		return choose_remove_vertex(v1,remove_v);
	}

	//check if v1 or v2 has been checked for exchanged
	int i;
	for (i=0; i<check_vertex_count;++i)
	{
		if(check_vertex[i]==v1)
		{
			add_v = v2;
			return choose_remove_vertex(v2,remove_v);
		}
		else if(check_vertex[i]==v2)
		{
			add_v = v1;
			return choose_remove_vertex(v1,remove_v);
		}
	}	

	
	int vertex_improvement_v1 = dscore[v1];
	int vertex_improvement_v2 = dscore[v2];

	int find_v1_adj=0;
	int find_v2_adj=0;
	if (vertex_improvement_v1+highest_dscore_c<=0)
		find_v1_adj = 1;
	if (vertex_improvement_v2+highest_dscore_c<=0)
		find_v2_adj = 1;

	if (find_v1_adj==1 && find_v2_adj==1)
	{
		return choose_exchange_pair_in_adj(v1,v2,add_v,remove_v);
	}

	//check both v1 and v2 for exchange
	int v = rng.next(v_num)+1;
	int init_v = v;

	if (find_v1_adj==1)
	{
		//init_v to v_num
		for (; v<=v_num;++v)
		{
			if(edge_weight[edge_between_vertices[v1][v]]!=0 && vertex_improvement_v1+edge_weight[edge_between_vertices[v1][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v1;
				remove_v = v;
				return 1;
			}

			if(vertex_improvement_v2+edge_weight[edge_between_vertices[v2][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v2;
				remove_v = v;
				return 1;
			}
		}
		//1 to init_v-1
		for (v=1; v<init_v;++v)
		{
			if(edge_weight[edge_between_vertices[v1][v]]!=0 && vertex_improvement_v1+edge_weight[edge_between_vertices[v1][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v1;
				remove_v = v;
				return 1;
			}

			if(vertex_improvement_v2+edge_weight[edge_between_vertices[v2][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v2;
				remove_v = v;
				return 1;
			}
		}
	}

	else if (find_v2_adj==1)
	{
		//init_v to v_num
		for (; v<=v_num;++v)
		{
			if(vertex_improvement_v1+edge_weight[edge_between_vertices[v1][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v1;
				remove_v = v;
				return 1;
			}

			if(edge_weight[edge_between_vertices[v2][v]]!=0 && vertex_improvement_v2+edge_weight[edge_between_vertices[v2][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v2;
				remove_v = v;
				return 1;
			}
		}
		//1 to init_v-1
		for (v=1; v<init_v;++v)
		{
			if(vertex_improvement_v1+edge_weight[edge_between_vertices[v1][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v1;
				remove_v = v;
				return 1;
			}

			if(edge_weight[edge_between_vertices[v2][v]]!=0 && vertex_improvement_v2+edge_weight[edge_between_vertices[v2][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v2;
				remove_v = v;
				return 1;
			}
		}
	}
	else
	{
		//init_v to v_num
		for (; v<=v_num;++v)
		{
			if(vertex_improvement_v1+edge_weight[edge_between_vertices[v1][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v1;
				remove_v = v;
				return 1;
			}

			if(vertex_improvement_v2+edge_weight[edge_between_vertices[v2][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v2;
				remove_v = v;
				return 1;
			}
		}
		//1 to init_v-1
		for (v=1; v<init_v;++v)
		{
			if(vertex_improvement_v1+edge_weight[edge_between_vertices[v1][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v1;
				remove_v = v;
				return 1;
			}
			if(vertex_improvement_v2+edge_weight[edge_between_vertices[v2][v]]+dscore[v]>0)
			{
				if(v_in_c[v]==0 || v==tabu_remove)continue;
				add_v = v2;
				remove_v = v;
				return 1;
			}
		}

	}

	check_vertex[check_vertex_count] = v1;
	check_vertex_count++;
	check_vertex[check_vertex_count] = v2;
	check_vertex_count++;

	return -1;

}


void update_edge_weight()
{
	Number* ptr_edgeNum = l_head;
	
	while (ptr_edgeNum->num!=e_num)
	{
		edge_weight[ptr_edgeNum->num]+= 1;

		dscore[edge[ptr_edgeNum->num].v1] += 1;
		dscore[edge[ptr_edgeNum->num].v2] += 1;
		
		conf_change[edge[ptr_edgeNum->num].v1] = 1;
		conf_change[edge[ptr_edgeNum->num].v2] = 1;

		ptr_edgeNum = ptr_edgeNum->next;		
	}
}


void cover_LS()
{
	int		add_v,remove_v;
	Number* ptr_edgeNum;
	Number* ptr_search_beg;
	int		exchang_flag;

	step = 0;
	ptr_search_beg = edge_ptr[e_num]->pre;
	while (step<max_steps)
	{
		exchang_flag = 0;

		//check the last uncovered edge in the list
		ptr_edgeNum = edge_ptr[e_num]->pre;
		if(choose_exchange_pair(edge[ptr_edgeNum->num].v1,edge[ptr_edgeNum->num].v2,add_v,remove_v)==1)
		{
			remove(remove_v);
			add(add_v);
			
			//tabu_add = remove_v;
			tabu_remove = add_v;
			++step;
			exchang_flag = 1;
			check_vertex_count = 0;

			ptr_search_beg = edge_ptr[e_num]->pre;
			if (ptr_search_beg!=NULL)ptr_search_beg = ptr_search_beg->pre;
		}

		//if (exchang_flag == 0)
		else
		{	
			//check from the last choose edge's pre edge to the head, finding an edge for exchange 
			for(ptr_edgeNum = ptr_search_beg; ptr_edgeNum!=NULL; ptr_edgeNum = ptr_edgeNum->pre)
			{
				if(choose_exchange_pair(edge[ptr_edgeNum->num].v1,edge[ptr_edgeNum->num].v2,add_v,remove_v)==1)
				{
					Number* np = ptr_edgeNum->next;
					while (edge[np->num].v1== add_v||edge[np->num].v2== add_v){np= np->next;}

					remove(remove_v);
					add(add_v);

					check_vertex_count = 0;

					//tabu_add = remove_v;
					tabu_remove = add_v;
					++step;
					exchang_flag = 1;

					ptr_search_beg = np->pre;
					break;
				}
			}
		}

		if(exchang_flag == 0)
		{
			update_edge_weight();

			/* random walk */
			int r = rng.next(l_size);
			ptr_edgeNum = l_head;
			while (r>0)
			{
				ptr_edgeNum = ptr_edgeNum->next;
				--r;
			}
			add_v = ( rng.next(2)==1) ? edge[ptr_edgeNum->num].v1 : edge[ptr_edgeNum->num].v2;

			do {remove_v = rng.next(v_num)+1;} while (v_in_c[remove_v]==0);

			remove(remove_v);
			add(add_v);
			
			//tabu_add = remove_v;
			tabu_remove = add_v;
			check_vertex_count = 0;
			++step;

			ptr_search_beg = edge_ptr[e_num]->pre;//after a random walk, we should search exchange from the end to head.
		}

		//update best solution if needed
		if (l_size == 0)
		{
			update_best_sol();
			
			if (c_size==optimal_size)
				return;
				
			update_target_size();
			
			if (l_size==0 && c_size==optimal_size)//update_target_size returns only when l_size>0 or finding optimal_size.
			{
				update_best_sol();
				return;
			}
		}

	}
}

int main(int argc, char* argv[])
{
	int myseed;
	char file_name[100];
	cout<<"EWCC"<<endl;
	if(argc < 2) {
    cout << "usage: " << argv[0] << " <random_seed_value>"<< endl;
    return -1;
  	}
	
	myseed = atoi(argv[1]);
	
	ifstream fin("parameter.txt");
	fin>>file_name;
	fin>>max_steps;
	
	fin>>optimal_size;
	fin.close();
    
    ifstream infile(file_name);
    if(infile==NULL)
    {
        cout<<"can't open instance file"<<endl;
        return -1;
    }

	cout<<"seed = "<<myseed<<endl;

	build_instance(infile);
	
	rng.seed(myseed);

	times(&start);

	init_sol();

	if(c_size + l_size > optimal_size ) 
		cover_LS();

	print_solution();

	times(&finish);
	double comp_time = double(finish.tms_utime - start.tms_utime + finish.tms_stime - start.tms_stime)/sysconf(_SC_CLK_TCK);
	comp_time = round(comp_time * 1000)/1000.0;
	cout<<"Program terminated in "<<comp_time<<" seconds "<<step<<" steps"<<endl;
	
	//check solution
	if(check_solution()==1)
		cout<<"the output solution is an indepence set."<<endl;
	else
		cout<<"the output solution is not an indepence set!"<<endl;

	destroy_instance();

	return 0;
}
