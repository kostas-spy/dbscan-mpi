//Kostas Spyropoulos - George Panou - University of Piraeus - 2019
/*2-axis DBSCAN Code with MPI*/

//Defines and includes
#define _CRT_SECURE_NO_WARNINGS
#include <limits.h>
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include <stdbool.h>

#define UNCLASSIFIED -1
#define NOISE -2
#define CORE_POINT 1
#define NOT_CORE_POINT 0
#define SUCCESS 0
#define FAILURE -3

//Structs of points, nodes and epsilon_neighbours
typedef struct point_s point_t;
struct point_s {
	double x, y;
	int cluster_id;
};

typedef struct cluster cluster_t;
struct cluster {
	double x, y;
	int gravity;
	int cluster_id;
	double radius;
	int points_count;
	bool exists;
};

typedef struct node_s node_t;
struct node_s {
	unsigned int index;
	node_t *next;
};

typedef struct epsilon_neighbours_s epsilon_neighbours_t;
struct epsilon_neighbours_s {
	unsigned int num_members;
	node_t *head, *tail;
};

//Functions Declaration
node_t *create_node(unsigned int index);
int append_at_end(unsigned int index, epsilon_neighbours_t *en);
epsilon_neighbours_t *get_epsilon_neighbours(unsigned int index, point_t *points, unsigned int num_points, double epsilon, double(*dist)(point_t *a, point_t *b));
void print_epsilon_neighbours(point_t *points, epsilon_neighbours_t *en);
void destroy_epsilon_neighbours(epsilon_neighbours_t *en);
unsigned int dbscan(point_t *points, unsigned int num_points, double epsilon, unsigned int minpts, double(*dist)(point_t *a, point_t *b));
int expand(unsigned int index, unsigned int cluster_id, point_t *points, unsigned int num_points, double epsilon, unsigned int minpts, double(*dist)(point_t *a, point_t *b));
int spread(unsigned int index, epsilon_neighbours_t *seeds, unsigned int cluster_id, point_t *points, unsigned int num_points, double epsilon, unsigned int minpts, double(*dist)(point_t *a, point_t *b));
double euclidean_dist(point_t *a, point_t *b);
void print_points(point_t *points, unsigned int num_points);
cluster_t *clusters_init(point_t *points, unsigned int num_points_per_proc, int clusters_count, int rank);
cluster_t *compare_clusters(cluster_t *clusters, cluster_t *send_cluster, point_t *points, point_t *recv_points, int cluster_count, int cluster_count_send, int minpts, double eps);
cluster_t *find_noise_relation(cluster_t *clusters, point_t *points, int num_points_per_proc, int c);

//Node Creation
node_t *create_node(unsigned int index)
{
	node_t *n = (node_t *)calloc(1, sizeof(node_t));
	if (n == NULL)
		perror("Failed to allocate node.");
	else {
		n->index = index;
		n->next = NULL;
	}
	return n;
}

//  Append At End
int append_at_end(unsigned int index, epsilon_neighbours_t *en)
{
	node_t *n = create_node(index);
	if (n == NULL) {
		free(en);
		return FAILURE;
	}
	if (en->head == NULL) {
		en->head = n;
		en->tail = n;
	}
	else {
		en->tail->next = n;
		en->tail = n;
	}
	++(en->num_members);
	return SUCCESS;
}

//Get Neighbours
epsilon_neighbours_t *get_epsilon_neighbours(unsigned int index, point_t *points, unsigned int num_points, double epsilon, double(*dist)(point_t *a, point_t *b))
{
	epsilon_neighbours_t *en = (epsilon_neighbours_t *)calloc(1, sizeof(epsilon_neighbours_t));
	if (en == NULL) {
		perror("Failed to allocate epsilon neighbours.");
		return en;
	}
	for (unsigned int i = 0; i < num_points; ++i) {
		if (i == index)
			continue;
		if (dist(&points[index], &points[i]) > epsilon)
			continue;
		else {
			if (append_at_end(i, en) == FAILURE) {
				destroy_epsilon_neighbours(en);
				en = NULL;
				break;
			}
		}
	}
	return en;
}

// Print Neighbours
void print_epsilon_neighbours(point_t *points, epsilon_neighbours_t *en)
{
	if (en) {
		node_t *h = en->head;
		while (h) {
			printf("(%lfm, %lf)\n", points[h->index].x, points[h->index].y);
			h = h->next;
		}
	}
}

// Destroy Neighbours
void destroy_epsilon_neighbours(epsilon_neighbours_t *en)
{
	if (en) {
		node_t *t, *h = en->head;
		while (h) {
			t = h->next;
			free(h);
			h = t;
		}
		free(en);
	}
}

// Main DBSCAN Algorithm
unsigned int dbscan(point_t *points, unsigned int num_points, double epsilon, unsigned int minpts, double(*dist)(point_t *a, point_t *b))
{
	unsigned int i, cluster_id = 0;
	for (i = 0; i < num_points; ++i) {
		if (points[i].cluster_id == UNCLASSIFIED) {
			if (expand(i, cluster_id, points, num_points, epsilon, minpts, dist) == CORE_POINT)
				++cluster_id;
		}
	}
	return cluster_id;
}

int expand(unsigned int index, unsigned int cluster_id, point_t *points, unsigned int num_points, double epsilon, unsigned int minpts, double(*dist)(point_t *a, point_t *b))
{
	int return_value = NOT_CORE_POINT;
	epsilon_neighbours_t *seeds = get_epsilon_neighbours(index, points, num_points, epsilon, dist);
	if (seeds == NULL)
		return FAILURE;

	if (seeds->num_members < minpts)
		points[index].cluster_id = NOISE;
	else {
		points[index].cluster_id = cluster_id;
		node_t *h = seeds->head;
		while (h) {
			points[h->index].cluster_id = cluster_id;
			h = h->next;
		}

		h = seeds->head;
		while (h) {
			spread(h->index, seeds, cluster_id, points, num_points, epsilon, minpts, dist);
			h = h->next;
		}

		return_value = CORE_POINT;
	}
	destroy_epsilon_neighbours(seeds);
	return return_value;
}

int spread(unsigned int index, epsilon_neighbours_t *seeds, unsigned int cluster_id, point_t *points, unsigned int num_points, double epsilon, unsigned int minpts, double(*dist)(point_t *a, point_t *b))
{
	epsilon_neighbours_t *spread = get_epsilon_neighbours(index, points, num_points, epsilon, dist);
	if (spread == NULL)
		return FAILURE;
	if (spread->num_members >= minpts) {
		node_t *n = spread->head;
		point_t *d;
		while (n) {
			d = &points[n->index];
			if (d->cluster_id == NOISE || d->cluster_id == UNCLASSIFIED) {
				if (d->cluster_id == UNCLASSIFIED) {
					if (append_at_end(n->index, seeds) == FAILURE) {
						destroy_epsilon_neighbours(spread);
						return FAILURE;
					}
				}
				d->cluster_id = cluster_id;
			}
			n = n->next;
		}
	}
	destroy_epsilon_neighbours(spread);
	return SUCCESS;
}

// Euclidean Distance
double euclidean_dist(point_t *a, point_t *b)
{
	return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
}

//Print Results
void print_points(point_t *points, unsigned int num_points)
{
	unsigned int i = 0;
	printf("Number of points: %u\n"
		" x     y     cluster_id\n"
		"-----------------------\n"
		, num_points);
	while (i < num_points) {
		printf("%5.2lf %5.2lf: %d\n", points[i].x, points[i].y, points[i].cluster_id);
		++i;
	}
	printf("\n");
}

//Calculate CenterOfMass per cluster, Radius and Count of points per cluster
cluster_t *clusters_init(point_t *points, unsigned int num_points_per_proc, int clusters_count, int rank)
{
	int i, k, j = 0;
	point_t clusterCOM, currentPoint, current_c_point;
	double *maxDistanceFromCOM = (double *)calloc(clusters_count, sizeof(double));
	cluster_t *clusters = (cluster_t *)calloc(clusters_count, sizeof(cluster_t));
	int *clusterPointsCount = (int *)calloc(clusters_count, sizeof(int));
	point_t *borderPoint = (point_t *)calloc(clusters_count, sizeof(point_t));

	for (i = 0; i < clusters_count; i++)
	{
		clusters[i].x = 0;
		clusters[i].y = 0;
		clusters[i].gravity = 1;
		clusters[i].exists = true;
		maxDistanceFromCOM[i] = 0;
		clusterPointsCount[i] = 0;
	}

	for (i = 0; i < num_points_per_proc; i++)
	{
		int currCluster = points[i].cluster_id;
		if (currCluster >= 0)
		{
			if (clusters[currCluster].x > points[i].x)
				clusters[currCluster].x = clusters[currCluster].x - (clusters[currCluster].x - points[i].x) / (clusters[currCluster].gravity);
			else if (clusters[currCluster].x < points[i].x)
				clusters[currCluster].x = clusters[currCluster].x + (points[i].x - clusters[currCluster].x) / (clusters[currCluster].gravity);

			if (clusters[currCluster].y > points[i].y)
				clusters[currCluster].y = clusters[currCluster].y - (clusters[currCluster].y - points[i].y) / (clusters[currCluster].gravity);
			else if (clusters[currCluster].y < points[i].y)
				clusters[currCluster].y = clusters[currCluster].y + (points[i].y - clusters[currCluster].y) / (clusters[currCluster].gravity);

			clusters[currCluster].gravity++;
		}
	}

	for (j = 0; j < num_points_per_proc; j++)
	{
		int currCluster = points[j].cluster_id;
		if (currCluster >= 0)
		{
			currentPoint.x = points[j].x;
			currentPoint.y = points[j].y;
			current_c_point.x = clusters[currCluster].x;
			current_c_point.y = clusters[currCluster].y;

			double currDist = euclidean_dist(&currentPoint, &current_c_point);
			if (currDist > maxDistanceFromCOM[currCluster]) {
				maxDistanceFromCOM[currCluster] = currDist;
				borderPoint[currCluster] = currentPoint;
			}
		}
	}

	if (clusters_count != 0)
	{
		for (i = 0; i < clusters_count; i++)
		{
			clusters[i].cluster_id = i;
			clusters[i].points_count = clusters[i].gravity - 1;
			clusters[i].radius = maxDistanceFromCOM[i];
			//printf("Cluster %d info: COM:(%lf,%lf), Count:%d, Radius:%lf from COM to point:(%lf,%lf), from rank %d\n", clusters[i].cluster_id, clusters[i].x, clusters[i].y, clusters[i].points_count, clusters[i].radius, borderPoint[i].x, borderPoint[i].y, rank);
		}
	}
	else
	{
		printf("Rank %d has no clusters", rank);
	}

	return clusters;
}

//Compare Clusters
cluster_t *compare_clusters(cluster_t *clusters, cluster_t *send_cluster, point_t *points, point_t *recv_points, int cluster_count, int cluster_count_send, int minpts, double eps)
{
	point_t a_1, a_2;
	int i = 0, index = 0, j = 0, c = 0, count_intersects = 0, points_c = 0, send_points_c = 0;
	double maxdist;
	bool flag = false;
	cluster_t *merged_cl = (cluster_t *)calloc(cluster_count_send + cluster_count, sizeof(cluster_t));
	for (i = 0; i < cluster_count_send + cluster_count; i++)
		merged_cl[i].exists = false;
	int *send_cluster_note = (int *)calloc(cluster_count_send, sizeof(int));
	for (i = 0; i < cluster_count_send; i++)
		send_cluster_note[i] = 0;

	for (index = 0; index < cluster_count; index++)
	{
		flag = false;
		for (i = 0; i < cluster_count_send; i++)
		{
			point_t *merged_cl_points = (point_t *)calloc(clusters[index].points_count + send_cluster[i].points_count, sizeof(point_t));

			a_1.x = clusters[index].x;
			a_1.y = clusters[index].y;
			a_2.x = send_cluster[i].x;
			a_2.y = send_cluster[i].y;

			if (fabs(clusters[index].radius - send_cluster[i].radius) >= euclidean_dist(&a_1, &a_2))  // one circle lies completely inside another
			{
				if (clusters[index].radius > send_cluster[i].radius)
				{
					printf("Cluster %d lies completely inside cluster %d\n", send_cluster[i].cluster_id, clusters[index].cluster_id);
					merged_cl[c].x = (clusters[index].x + send_cluster[i].x) / 2;
					merged_cl[c].y = (clusters[index].y + send_cluster[i].y) / 2;
					merged_cl[c].gravity = clusters[index].gravity + send_cluster[i].gravity;
					merged_cl[c].cluster_id = c;
					merged_cl[c].radius = clusters[index].radius;
					merged_cl[c].points_count = clusters[index].points_count + send_cluster[i].points_count;
					merged_cl[c].exists = true;
					c++;
					flag = true;
				}
				else
				{
					printf("Cluster %d lies completely inside cluster %d\n", clusters[index].cluster_id, send_cluster[i].cluster_id);
					merged_cl[c].x = (clusters[index].x + send_cluster[i].x) / 2;
					merged_cl[c].y = (clusters[index].y + send_cluster[i].y) / 2;
					merged_cl[c].gravity = clusters[index].gravity + send_cluster[i].gravity;
					merged_cl[c].cluster_id = c;
					merged_cl[c].radius = send_cluster[i].radius;
					merged_cl[c].points_count = clusters[index].points_count + send_cluster[i].points_count;
					merged_cl[c].exists = true;
					c++;
					flag = true;
				}
			}
			else if (fabs(clusters[index].radius - send_cluster[i].radius) < euclidean_dist(&a_1, &a_2)
				&& (clusters[index].radius + send_cluster[i].radius) > euclidean_dist(&a_1, &a_2))  // intersect in two points
			{
				point_t curr_c_pnt, currpnt;
				curr_c_pnt.x = (send_cluster[i].x + clusters[index].x) / 2;
				curr_c_pnt.y = (send_cluster[i].y + clusters[index].y) / 2;
				maxdist = 0;
				for (int j = 0; j < clusters[index].points_count; j++)
				{
					if ((euclidean_dist(&points[j + points_c], &a_2) <= send_cluster[i].radius))
					{
						count_intersects++;
					}
					currpnt.x = clusters[index].x;
					currpnt.y = clusters[index].y;
					if (euclidean_dist(&currpnt, &curr_c_pnt) >= maxdist)
						maxdist = euclidean_dist(&currpnt, &curr_c_pnt);
				}

				for (int j = 0; j < send_cluster[i].points_count; j++)
				{
					if ((euclidean_dist(&recv_points[j + send_points_c], &a_1) <= clusters[index].radius))
					{
						count_intersects++;
					}
					currpnt.x = send_cluster[i].x;
					currpnt.y = send_cluster[i].y;
					if (euclidean_dist(&currpnt, &curr_c_pnt) >= maxdist)
						maxdist = euclidean_dist(&currpnt, &curr_c_pnt);
				}

				if (count_intersects > 0)
				{
					merged_cl[c].x = (clusters[index].x + send_cluster[i].x) / 2;
					merged_cl[c].y = (clusters[index].y + send_cluster[i].y) / 2;
					merged_cl[c].gravity = clusters[index].gravity + send_cluster[i].gravity;
					merged_cl[c].cluster_id = c;
					merged_cl[c].points_count = clusters[index].points_count + send_cluster[i].points_count;
					merged_cl[c].radius = maxdist;
					merged_cl[c].exists = true;
					c++;
					flag = true;
					printf("Cluster %d intersects in two points with cluster %d and the intersected points are %d\n",
						send_cluster[i].cluster_id, clusters[index].cluster_id, count_intersects);
				}
				else
				{
					printf("Cluster %d intersects in two points with cluster %d but have no intersected points and the clusters don't merge\n",
						send_cluster[i].cluster_id, clusters[index].cluster_id);
					send_cluster_note[i] = send_cluster_note[i] + 1;
				}
			}
			else if ((clusters[index].radius + send_cluster[i].radius) == euclidean_dist(&a_1, &a_2))  // intersect in one point
			{
				point_t curr_c_pnt, currpnt, int_pnt;
				curr_c_pnt.x = (send_cluster[i].x + clusters[index].x) / 2;
				curr_c_pnt.y = (send_cluster[i].y + clusters[index].y) / 2;
				maxdist = 0;
				for (int j = 0; j < clusters[index].points_count; j++)
				{
					if ((euclidean_dist(&points[j + points_c], &a_2) == send_cluster[i].radius))
					{
						int_pnt = points[j + points_c];
						count_intersects++;
					}
					currpnt.x = clusters[index].x;
					currpnt.y = clusters[index].y;
					if (euclidean_dist(&currpnt, &curr_c_pnt) >= maxdist)
						maxdist = euclidean_dist(&currpnt, &curr_c_pnt);
				}

				for (int j = 0; j < send_cluster[i].points_count; j++)
				{
					if ((euclidean_dist(&recv_points[j + send_points_c], &a_1) == clusters[index].radius))
					{
						int_pnt = recv_points[j + send_points_c];
						count_intersects++;
					}
					currpnt.x = send_cluster[i].x;
					currpnt.y = send_cluster[i].y;
					if (euclidean_dist(&currpnt, &curr_c_pnt) >= maxdist)
						maxdist = euclidean_dist(&currpnt, &curr_c_pnt);
				}

				if (count_intersects > 0)
				{
					merged_cl[c].x = (clusters[index].x + send_cluster[i].x) / 2;
					merged_cl[c].y = (clusters[index].y + send_cluster[i].y) / 2;
					merged_cl[c].gravity = clusters[index].gravity + send_cluster[i].gravity;
					merged_cl[c].cluster_id = c;
					merged_cl[c].points_count = clusters[index].points_count + send_cluster[i].points_count;
					merged_cl[c].radius = maxdist;
					merged_cl[c].exists = true;
					c++;
					flag = true;
					printf("Cluster %d intersects in one point with cluster %d and the intersected point is (%lf,%lf)\n",
						send_cluster[i].cluster_id, clusters[index].cluster_id, int_pnt.x, int_pnt.y);
				}
				else
				{
					printf("Cluster %d intersects in one point with cluster %d but the intersected point doesn't belong in the dataset, so the clusters don't merge\n",
						send_cluster[i].cluster_id, clusters[index].cluster_id);
					send_cluster_note[i] = send_cluster_note[i] + 1;
				}
			}
			else  // do not intersect
			{
				printf("Clusters %d and %d have nothing in common\n", send_cluster[i].cluster_id, clusters[index].cluster_id);
				send_cluster_note[i] = send_cluster_note[i] + 1;
			}
			send_points_c = send_points_c + send_cluster[i].points_count;
		}
		//printf("\n\n");
		if (!flag)
		{
			merged_cl[c].x = clusters[index].x;
			merged_cl[c].y = clusters[index].y;
			merged_cl[c].gravity = clusters[index].gravity;
			merged_cl[c].cluster_id = c;
			merged_cl[c].points_count = clusters[index].points_count;
			merged_cl[c].radius = clusters[index].radius;
			merged_cl[c].exists = true;
			c++;
		}
		points_c = points_c + clusters[index].points_count;
	}
	for (i = 0; i < cluster_count_send; i++)
	{
		if (send_cluster_note[i] == cluster_count)
		{
			merged_cl[c].x = send_cluster[i].x;
			merged_cl[c].y = send_cluster[i].y;
			merged_cl[c].gravity = send_cluster[i].gravity;
			merged_cl[c].cluster_id = c;
			merged_cl[c].points_count = send_cluster[i].points_count;
			merged_cl[c].radius = send_cluster[i].radius;
			merged_cl[c].exists = true;
			c++;
		}
	}
	printf("No of cluster after merge: %d\n", c);
	return merged_cl;
}

cluster_t *find_noise_relation(cluster_t *clusters, point_t *points, int num_points_per_proc, int c)
{
	int i,j;
	point_t merged_c;
	for (i = 0; i < num_points_per_proc * 2; i++)
	{
		if (points[i].cluster_id == -2)
		{
			for (j = 0; j < c; j++)
			{
				merged_c.x = clusters[j].x;
				merged_c.y = clusters[j].y;
				if (euclidean_dist(&points[i], &merged_c) <= clusters[j].radius)
				{
					clusters[j].points_count += 1;
					points[i].cluster_id = clusters[j].cluster_id;
				}
			}
		}
	}
	return clusters;
}

void print_cluster_info(int clusters_count, cluster_t * clusters, int rank, point_t * points)
{
	int pts_sum = 0, j = 0, jj = 0;
	if (clusters_count != 0)
	{
		for (int i = 0; i < clusters_count; i++)
		{
			printf("Cluster %d info: COM:(%lf,%lf), Count:%d, Radius:%lf, rank: %d\n", clusters[i].cluster_id, clusters[i].x, clusters[i].y, clusters[i].points_count, clusters[i].radius, rank);
			for (j = 0; j < clusters[i].points_count; j++)
			{
				jj = j;
				while (points[jj + pts_sum].cluster_id == -2)
				{
					jj++;
				}
				printf("Cluster %d points: x=%lf, y=%lf\n", clusters[i].cluster_id, points[jj + pts_sum].x, points[jj + pts_sum].y);
			}
			pts_sum = pts_sum + clusters[i].points_count + jj;
		}
	}
	else
		printf("Rank %d has no clusters", rank);
}

//Main
int main(int argc, char *argv[])
{
	point_t *points;
	double epsilon;
	unsigned int minpts, num_points, num_points_per_proc, i = 0, k = 0, j = 0;
	int rank, size, clusters_count = 0;
	cluster_t *clusters;
	clock_t t;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (rank == 0)
	{
		t = clock();
		printf("Begin Time Calculation...\n");

		FILE *file = fopen("input.txt", "r+");
		fscanf(file, "%lf %u %u\n", &epsilon, &minpts, &num_points);
		printf("Epsilon: %lf\n", epsilon);
		printf("Minimum points: %u\n", minpts);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != 0)
	{
		int start, end;
		double hi, psi;
		FILE *file = fopen("input.txt", "r+");
		fscanf(file, "%lf %u %u\n", &epsilon, &minpts, &num_points);
		num_points_per_proc = num_points / (size - 1);

		point_t *p = (point_t *)calloc(num_points_per_proc, sizeof(point_t));
		if (p == NULL) {
			perror("Failed to allocate points array");
			return 0;
		}

		start = (num_points_per_proc * (rank - 1));
		if (rank == size - 1)
			end = num_points;
		else
			end = num_points_per_proc * rank;

		while (i < num_points) {
			if (i == start)
				while (i < end)
				{
					fscanf(file, "%lf %lf\n", &(p[j].x), &(p[j].y));
					p[j].cluster_id = UNCLASSIFIED;
					++i;
					++j;
				}
			fscanf(file, "%lf %lf\n", &hi, &psi);
			++i;
		}

		points = p;
		fclose(file);

		if (num_points_per_proc)
		{
			clusters_count = dbscan(points, num_points_per_proc, epsilon, minpts, euclidean_dist);
			//print_points(points, num_points_per_proc);
		}

			/* create a type for struct points_t */
			const int	 nitems_points = 3;
			int          blocklengths_points[3] = { 1,1,1 };
			MPI_Datatype types_points[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
			MPI_Datatype mpi_points_type;
			MPI_Aint     offsets_points[3];

			offsets_points[0] = offsetof(point_t, x);
			offsets_points[1] = offsetof(point_t, y);
			offsets_points[2] = offsetof(point_t, cluster_id);

			MPI_Type_create_struct(nitems_points, blocklengths_points, offsets_points, types_points, &mpi_points_type);
			MPI_Type_commit(&mpi_points_type);
			/* create a type for struct clusters */
			const int	 nitems_cl = 7;
			int          blocklengths_cl[7] = { 1,1,1,1,1,1,1 };
			MPI_Datatype types_cl[7] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT, MPI_C_BOOL };
			MPI_Datatype mpi_cl_type;
			MPI_Aint     offsets_cl[7];

			offsets_cl[0] = offsetof(cluster_t, x);
			offsets_cl[1] = offsetof(cluster_t, y);
			offsets_cl[2] = offsetof(cluster_t, gravity);
			offsets_cl[3] = offsetof(cluster_t, cluster_id);
			offsets_cl[4] = offsetof(cluster_t, radius);
			offsets_cl[5] = offsetof(cluster_t, points_count);
			offsets_cl[6] = offsetof(cluster_t, exists);

			MPI_Type_create_struct(nitems_cl, blocklengths_cl, offsets_cl, types_cl, &mpi_cl_type);
			MPI_Type_commit(&mpi_cl_type);

		//printf("Clusters of rank %d\n", rank);
		clusters = clusters_init(points, num_points_per_proc, clusters_count, rank);

		int n = 2, m = 0, c = 0;
		while (m <= log(size - 1))
		{
			if (rank % n == n / 2)
			{
				MPI_Send(&clusters_count, 1, MPI_INT, rank + n / 2, 10, MPI_COMM_WORLD);
				MPI_Send(points, num_points_per_proc, mpi_points_type, rank + n / 2, 20, MPI_COMM_WORLD);
				MPI_Send(clusters, clusters_count, mpi_cl_type, rank + n / 2, 30, MPI_COMM_WORLD);

				//printf("Rank %d: sent struct clusters to rank %d\n", rank, rank + n / 2);
			}
			else if (rank % n == 0)
			{
				int recv_cluster_count, no_of_intersections;
				c = 0;
				MPI_Status status;

				MPI_Recv(&recv_cluster_count, 1, MPI_INT, rank - n / 2, 10, MPI_COMM_WORLD, &status);
				cluster_t *recv_cluster = (cluster_t *)calloc(recv_cluster_count, sizeof(cluster_t));
				point_t *recv_points = (point_t *)calloc(num_points_per_proc, sizeof(point_t));
				MPI_Recv(recv_points, num_points_per_proc, mpi_points_type, rank - n / 2, 20, MPI_COMM_WORLD, &status);
				MPI_Recv(recv_cluster, recv_cluster_count, mpi_cl_type, rank - n / 2, 30, MPI_COMM_WORLD, &status);
				
				/*print_cluster_info(clusters_count, clusters, rank, points);
				printf("\n");
				print_cluster_info(recv_cluster_count, recv_cluster, rank - n / 2, recv_points);
				printf("\n\n");*/
				
				cluster_t *merged_cl = (cluster_t *)calloc(recv_cluster_count + clusters_count, sizeof(cluster_t));
				merged_cl = compare_clusters(clusters, recv_cluster, points, recv_points, clusters_count, recv_cluster_count, minpts, epsilon);
				for (i = 0; i < recv_cluster_count + clusters_count; i++)
				{
					if (merged_cl[i].exists == true)
					{
						c += 1;
					}
				}

				clusters = merged_cl;
				point_t *merged_pts = (point_t *)calloc(num_points_per_proc * 2, sizeof(point_t));
				for (i = 0; i < num_points_per_proc * 2; i++)
				{
					if (i < num_points_per_proc)
						merged_pts[i] = points[i];
					else
						merged_pts[i] = recv_points[i - num_points_per_proc];
				}
				points = merged_pts;
				clusters_count = c;

				merged_cl = find_noise_relation(clusters, points, num_points_per_proc, c);

				printf("Merge of ranks %d and %d\n", rank, rank - n / 2);
				for (i = 0; i < c; i++)
				{
					printf("Merged Cluster: %d, com:(%lf,%lf), radius:%lf, CountOfPoints:%d\n",
						merged_cl[i].cluster_id, merged_cl[i].x, merged_cl[i].y, merged_cl[i].radius, merged_cl[i].points_count);
				}
				printf("\n\n");
			}
			n *= 2;	m += 1;
		}
		free(points);
		free(clusters);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0)
	{
		t = (clock() - t);
		float timeSpentQS = ((float)t) / CLOCKS_PER_SEC;
		printf("It took me %d clicks to calculate DBSCAN using %u processors (%.3f seconds).\n", t, size, timeSpentQS);
	}

	MPI_Finalize();
	return 0;
}
