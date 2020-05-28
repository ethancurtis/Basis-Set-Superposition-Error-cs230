#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_ELEM 94 // largest atomic element number
#define AU_TO_ANG 0.5291772083 



struct Mol {
	double * x;
	double * y;
	double * z;
	double * xva;
	int * atoms;
	int n;
};

struct Params {
	double * p;
	double * za;
	double * emiss;
	int * nbas;
	int damp;
	int base;
	int b973c;
	int echo;	
	double dmp_scal;
	double dmp_exp;
	int grad;
};

void readXYZ(char *, struct Mol *);
double aTof(char *s);
void testgCP();
void readParams(char * filename, struct Params * params);

double gcp_egrad(struct Mol *, struct Params *);
void srb_egrad(struct Mol *, double rscal, double qscal, double * e, double g[][3]);
double ssovl(double r, int iat, int jat, int * iz, double za, double zb); 
void setzet(double eta, double * za);
void setParams(struct Params *, char * methName);
void paramGrad(struct Params *, struct Mol *, double  *, double eps, int doEmiss, int doZa);
void setXva(struct Params * params, struct Mol * mol);
void freeMol(struct Mol * mol);

double a0(double);
double a1(double);
double a2(double);
double a3(double);
double a4(double);
double a5(double);
double a6(double);
double b0(double);
double b1(double);
double b2(double);
double b3(double);
double b4(double);
double b5(double);
double b6(double);
double bint(double, int);
int fact(int);
double g1s1s(double za,double zb,double fa,double fb,double r,int sameEl);
double g2s1s(double za,double zb,double fa,double fb,double r,int sameEl);
double g2s2s(double za,double zb,double fa,double fb,double r,int sameEl);
double g2s3s(double za,double zb,double fa,double fb,double r,int sameEl);
double g3s1s(double za,double zb,double fa,double fb,double r,int sameEl);
double g3s2s(double za,double zb,double fa,double fb,double r,int sameEl);
double g3s3s(double za,double zb,double fa,double fb,double r,int sameEl);
 
int main(int argc, char *argv[]) {
	struct Mol mol;
	struct Params params;
	setParams(&params, "hf/minis");
	char * filename = argv[1];
	readParams(filename, &params);	
	double *pgrad = (double *)malloc(sizeof(double) * 4);
	int i;
	for (i = 2; i < argc; i++) {
		readXYZ(argv[i], &mol);
		setXva(&params, &mol);
		//paramGrad(&params, &mol, pgrad, 0.0001, 0, 0);
		printf("%f\n",gcp_egrad(&mol, &params));
		freeMol(&mol);
	} 
	//testgCP();
}

//reads in parameters, calls necessary functions, etc.
/*
int do_gcp(int* dograd, int n, double* xyz, int* iz, double* gcpgrad, double* gcpenergy, char* method, int methodLength) {
	

	load_params();
	if (param_file) {
		read_params();
	}
}
*/

void testgCP() {
	size_t n = 2;
	double x[2], y[2], z[2];
	x[0] = x[1] = y[0] = y[1] = z[0] = z[1] = 0;
	// number of basis functions for 6-31gs
	int iz[2] = {1, 1};
	// double p[4] = {0.1724, 1.2804, 0.8568, 1.2342};
	//
	// Below: params for dft/631gd
	double dist[3] = {1.0, 3.0, 5.0};
	double gcp;
	double xva[2] = {1.5, 1.5};
	int el[3] = {2, 6, 16};
	int el2[3] = {1, 8, 14};
	int i,j,k;
	struct Mol mol;
	mol.n = n;
	mol.x = x;
	mol.y = y;
	mol.z = z;
	mol.atoms = iz;
	mol.xva = xva;
	struct Params params;
	setParams(&params, "dft/631gd");
	for (i = 0; i < 3; i++) {
		iz[0] = el[i];
		for (k = 0; k < 3; k++) {
			mol.x[0] = dist[k];
			for (j = 0; j < 3; j++) {
				iz[1] = el[j];
				setXva(&params, &mol);
				printf("Molecule: %d %d distance: %f test gcp: %e\n",iz[0],iz[1],dist[k],gcp_egrad(&mol, &params));
			}
			iz[1] = el2[i];	
			setXva(&params, &mol);
			printf("Molecule: %d %d distance: %f test gcp: %e\n",iz[0],iz[1],dist[k],gcp_egrad(&mol, &params));
		}
	}
}

void setParams(struct Params * params, char * methName) {
	int i;
	if(!strcmp(methName,"dft/631gd")) {
		double p[4] = {0.3405, 1.6127, 0.8589, 1.2830};
		int nbas[17] = {-1, 2, 5, 14, 14, 14, 14, 14, 14, 14, 14, 18, 18, 18, 18, 18, 18};
		double emiss[17] = {0, 0.010083, 0.008147, 0, 0, 0, 0.021407, 0, 0.036746, 0, 0, 0, 0, 0, 0.072534, 0, 0.056408 };
		double za[37];
		setzet(p[1],za);
		params->p = (double *) malloc(sizeof(double) * 4);
		for (i=0;i<4;i++) {
			params->p[i] = p[i];
		}
		params->nbas = (int *)malloc(sizeof(int) * 17);
		for (i=0;i<17;i++) {
			params->nbas[i] = nbas[i];
		} 
		params->emiss = (double *)malloc(sizeof(double) * 17);
		for (i=0;i<17;i++) {
			params->emiss[i] = emiss[i];
		}
		params->za = (double *) malloc(sizeof(double) * 37);
		for (i=0; i<37; i++) {
			params->za[i] = za[i];
		}
	}
	else if(!strcmp(methName,"hf/minis")) {
		double p[4] = {0.1290, 1.1526, 1.1549, 1.1763};
		int nbas[18] = {-1, 1, 1, 2, 2, 5, 5, 5, 5, 5, 5, 6, 6, 9, 9, 9, 9, 9};
		double emiss[18] = {0, 0.042400, 0.028324, 0.252661, 0.197201, 0.224237, 0.279950, 0.357906, 0.479012, 0.638518, 0.832349, 1.232920, 1.343390, 1.448280, 1.613360, 1.768140, 1.992010, 2.233110};
		double za[37];
		setzet(p[1],za);
		params->p = (double *) malloc(sizeof(double) * 4);
		for (i=0;i<4;i++) {
			params->p[i] = p[i];
		}
		params->nbas = (int *)malloc(sizeof(int) * 18);
		for (i=0;i<18;i++) {
			params->nbas[i] = nbas[i];
		} 
		params->emiss = (double *)malloc(sizeof(double) * 18);
		for (i=0;i<18;i++) {
			params->emiss[i] = emiss[i];
		}
		params->za = (double *) malloc(sizeof(double) * 37);
		for (i=0; i<37; i++) {
			params->za[i] = za[i];
		}
	}
	params->damp = 0;
	params->echo = 0;
	params->base = 1;
	params->b973c = 0;
	params->dmp_scal = 1;
	params->dmp_exp = 0;
	params->grad = 0;
}
		
void readParams(char * filename, struct Params * params) {
	char line[1000];
	FILE *paramFile = fopen(filename, "r");
	int counter = 0;
	int i;
	while (fgets(line, 1000, paramFile) != NULL) {
		if (counter == 0) {
			params->p[0] = aTof(strtok(line," "));
			for (i = 1; i < 4; i++) {
				params->p[i] = aTof(strtok(NULL," "));
			}
		}
		if (counter == 1) {}
		if (counter == 2) {}
		counter++;
	}
	fclose(paramFile);
	setzet(params->p[1],params->za);
}

void readXYZ(char * filename, struct Mol * mol) {
	FILE *xyzFile = fopen(filename, "r");
	char line[100];
	fgets(line, 100, xyzFile);
	int numAtoms = atoi(line);
	char line2[100];
	fgets(line2,100,xyzFile);
	double x[numAtoms];
	double y[numAtoms];
	double z[numAtoms];
	int atoms[numAtoms];
	int i;
	for (i = 0; i < numAtoms; i++) {
		fgets(line, 100, xyzFile);
		char * atomString = strtok(line, "\t");
		if (!strcmp(atomString,"H")) {
			atoms[i] = 1;
		}
		else if (!strcmp(atomString,"C")) {
			atoms[i] = 6;
		}
		else if (!strcmp(atomString,"N")) {
			atoms[i] = 7;
		}
		else if (!strcmp(atomString,"O")) {
			atoms[i] = 8;
		}
		else if (!strcmp(atomString,"F")) {
			atoms[i] = 9;
		}
		else if (!strcmp(atomString,"Cl")) {
			atoms[i] = 17;
		}
		else if (!strcmp(atomString,"S")) {
			atoms[i] = 16;
		}
		else {
			printf("ATOM %s UNRECOGNIZED ATOM TYPE in position %d\n", atomString, i);
		}
		x[i] = aTof(strtok(NULL, "\t"));
		y[i] = aTof(strtok(NULL, "\t"));
		z[i] = aTof(strtok(NULL, "\t"));
	}
	fclose(xyzFile);
	mol->x = (double *) malloc(sizeof(double) * numAtoms);
	memcpy(mol->x,x,numAtoms*sizeof(double));
	mol->y = (double *) malloc(sizeof(double) * numAtoms);
	memcpy(mol->y,y,numAtoms*sizeof(double));
	mol->z = (double *) malloc(sizeof(double) * numAtoms);
	memcpy(mol->z,z,numAtoms*sizeof(double));
	mol->atoms = (int *) malloc(sizeof(int) * numAtoms);
	memcpy(mol->atoms,atoms,numAtoms*sizeof(int));
	mol->n = numAtoms;
}

double aTof(char *s) {
    double val, power;                      
    int i, sign, expSign, exp;              
                                            
    for (i = 0; isspace(s[i]); i++)         
        ;                                   
    sign = 1;                               
    if (s[i] == '-') {                      
        sign = -1;                          
        i++;                                
    }                                       
    else if (s[i] == '+') {                 
        i++;                                
    }                                       
    for (val = 0.0; isdigit(s[i]); i++) {   
        val = 10.0 * val + (s[i] - '0');    
    }                                       
    if (s[i] == '.') {                      
        i++;                                
    }                                       
    for (power = 1.0; isdigit(s[i]); i++) { 
        val = 10.0 * val + (s[i] - '0');    
        power *= 10;                        
    }                                       
    if (s[i] == 'e' || s[i] == 'E') {       
        i++;                                
    }                                       
    expSign = 1;                            
    if (s[i] == '-') {                      
        expSign = -1;                       
        i++;                                
    }                                       
    else if (s[i] == '+') {                 
        i++;                                
    }                                       
    for (exp = 0; isdigit(s[i]); i++) {     
        exp = 10 * exp + (s[i] - '0');      
    }                                       
    if (expSign == 1) {                     
        for (; exp > 0; exp--) {            
            power /= 10;                    
        }                                   
    }                                       
    else {                                  
        for (; exp > 0; exp--) {            
            power *= 10;                    
        }                                   
    }                                       
    return sign * val / power;              
}                                           
	
void paramGrad(struct Params * params, struct Mol * mol, double * grad, double eps, int doEmiss, int doZa) {
	int i;
	double gcpUp, gcpDown;
	for (i = 0; i < 4; i++) {
		params->p[i] += eps;
		setzet(params->p[1], params->za);
		gcpUp = gcp_egrad(mol, params);
		params->p[i] -= 2 * eps;
		setzet(params->p[1], params->za);
		gcpDown = gcp_egrad(mol, params);
		params->p[i] += eps;
		grad[i] = (gcpUp - gcpDown) / (2 * eps);
		//printf("gcpUp %f gcpDown %f grad %f i %d\n",gcpUp, gcpDown, grad[i], i);
		printf("%f ", grad[i]);
	}
	printf("\n");
	if (doEmiss) {
	}
	if (doZa) {
	}
}
		
void setXva(struct Params * params, struct Mol * mol) {
	int i;
	double xva[mol->n];
	for (i = 0; i < mol->n; i++) {
		xva[i] = params->nbas[mol->atoms[i]] - mol->atoms[i] / 2.0;
	}
	mol->xva = (double *) malloc(mol->n * sizeof(double));
	memcpy(mol->xva, xva, mol->n * sizeof(double));
}	
	
void freeMol(struct Mol * mol) {
	free(mol->x);
	free(mol->y);
	free(mol->z);
	free(mol->atoms);
	free(mol->xva);
}

// calculates the gcp correction
double gcp_egrad(struct Mol * mol, struct Params * params) {
	int np, iat, jat; 
	double tg, t0, t1;
	double dum22;
	double va, vb;
	double r, dx, dy, dz, rscal, rscalexp, rscalexpml, r0abij;
	double r0ab[MAX_ELEM + 1][MAX_ELEM + 1];
	double tmp[3], ecp, dum, tmpa, tmpb, dum2, gs[3];
	double sab, p0, p1, p2, p3, ene_old, ene_old_num, ene_old_den;
	double dampval, ene_damp, grd_damp;
	double thrR = 60; // 60 bohr cutoff

	// unpack mol struct
	int n = mol->n;

	// unpack params struct
	int grad = params->grad;
	int echo = params->echo;
	int damp = params->damp;
	int base = params->base;
	int b973c = params->b973c;
	double dmp_scal = params->dmp_scal;
	double dmp_exp = params->dmp_exp; 
	
	double ebas, g[n][3], ea, gab, gcp;

	dum = ecp = tg = ebas = gcp = 0;
	int i;
	// initialize test r0ab matrix
	r0ab[1][2] = r0ab[2][1] = 3.1613 / AU_TO_ANG;
	r0ab[2][2] = 3.3605 / AU_TO_ANG;
	r0ab[2][6] = r0ab[6][2] = 2.8674 / AU_TO_ANG;
	r0ab[2][16] = r0ab[16][2] = 3.1866 / AU_TO_ANG;
	r0ab[6][6] = 3.3547 / AU_TO_ANG;
	r0ab[6][8] = r0ab[8][6] = 3.9837 / AU_TO_ANG;
	r0ab[6][16] = r0ab[16][6] = 3.6860 / AU_TO_ANG;
	r0ab[14][16] = r0ab[16][14] = 3.4228 / AU_TO_ANG;

	/*
	for (i = 0; i < 4; i ++) {
		printf("p[%d] = %f\n",i, params->p[i]);
	}
	for (i = 0; i < 18; i++) {
		printf("za[%d] = %f\n",i,params->za[i]);
	}
	*/

	if (damp) {
		printf("damping!");
	}

	if (b973c) {
	//	B97-3c case: no GCP, modified SRB
		thrR = 30; // 30 bohr cutoff
		srb_egrad(mol, 10, 0.08, &ebas, g);
		gcp += ebas;
		printf("%18.10f (a.u.) || %11.4f (kcal/mol)\n",ebas, ebas * 627.5099); 
	}
	else {
	//  standard, do GCP
		if (echo) {
			printf("cutoffs: %5.1f distance (bohr)\n%8.1e noise (a.u.)\n%d SR-damping\n",thrR, DBL_EPSILON,damp);
		}
		p0 = fabs(params->p[0]);
		p1 = fabs(params->p[1]);
		p2 = fabs(params->p[2]);
		p3 = fabs(params->p[3]);
		
//		TODO: implement setr0ab
//		setr0ab(r0ab); // get cutoff radii for all element pairs
	
		if(echo) {
			//printf
		}
		// Loop over atoms i
		for (iat = 0; iat < n; iat++) {
			va = mol->xva[iat];
			ea = 0;
			np = 0;
			// calculate the BSSE due to atom jat by looping over all atoms j
			for (jat = iat + 1; jat < n; jat++) {
				dx = mol->x[iat] - mol->x[jat];
				dy = mol->y[iat] - mol->y[jat];
				dz = mol->z[iat] - mol->z[jat];
				r = sqrt(dx * dx + dy * dy + dz * dz) / AU_TO_ANG;
				vb = mol->xva[jat]; // # of bf available from jat
				if (vb < 0.5 || r > thrR) { // distance threshold
					continue;
				}
				//calculate Slater overlap sab
				sab = ssovl(r, iat, jat, mol->atoms, params->za[mol->atoms[iat]], params->za[mol->atoms[jat]]); 
				if (fabs(sab) < DBL_EPSILON) { // noise cutoff for sab
					continue;
				}
				// evaluate gcp expression for pair ij
				ene_old_num = exp(-p2 * pow(r, p3));
				ene_old_den = sqrt(vb * sab);
				ene_old = ene_old_num / ene_old_den;
				if (fabs(ene_old) < DBL_EPSILON) { // noise cutoff for damp
					continue;
				}
				if (damp) {
					// D3 r0ab radii
					r0abij = r0ab[mol->atoms[iat]][mol->atoms[jat]];
					rscal = r / r0abij;
					// covalent radii
					rscalexp = pow(rscal, dmp_exp);
					dampval = 1.0 - 1.0 / (1 + dmp_scal * rscalexp);
					ene_damp = ene_old * dampval;
					ea += params->emiss[mol->atoms[iat]] * ene_damp;
				}
				else {
					ea += params->emiss[mol->atoms[iat]] * ene_old;
					//printf("i, j = %d, %d, overlap = %f, ea = %f, contribution = %f\n", iat, jat, sab, ea, params->emiss[mol->atoms[iat]] * ene_old);	
				}
				// evaluate gcp expression for pair ji
				ene_old_den = sqrt(va * sab);
				ene_old = ene_old_num / ene_old_den;
				if (fabs(ene_old) < DBL_EPSILON) { // noise cutoff for damp
					continue;
				}
				if (damp) {
					// D3 r0ab radii
					r0abij = r0ab[mol->atoms[iat]][mol->atoms[jat]];
					rscal = r / r0abij;
					// covalent radii
					rscalexp = pow(rscal, dmp_exp);
					dampval = 1.0 - 1.0 / (1 + dmp_scal * rscalexp);
					ene_damp = ene_old * dampval;
					ea += params->emiss[mol->atoms[jat]] * ene_damp;
				}
				else {
					ea += params->emiss[mol->atoms[jat]] * ene_old;
					//printf("i, j = %d, %d, overlap = %f, ea = %f, contribution = %f\n", jat, iat, sab, ea, params->emiss[mol->atoms[jat]] * ene_old);	
				}
				np+=2; //  site counter (# atoms contributing to the 'atomic bsse')
/*
				if (grad) { //gradient for the i,j pair
					//gab = gsovl(r, iat, jat, iz, za[iz[iat]], zb[iz[jat]]);
					//TODO : implement gab
					//TODO : need to make 2 copies of this section, one for ij pair and one for ji pair
					gs[0] = gab * dx;
					gs[1] = gab * dy;
					gs[2] = gab * dz;
					dum = exp(-p2 * pow(r, p3));
					dum2 = 2 * p2 * p3 * pow(r, p3) * sab / r;
					dum22 = r * pow(sab, 1.5);
					tmpb = dum22 * sqrt(vb);
					if (damp) {
						rscalexpml = pow(rscal, dmp_exp - 1);
						grd_damp = (dmp_scal * dmp_exp * rscalexpml / r0abij) / ((dmp_scal * rscalexp + 1) * (dmp_scal * rscalexp + 1));
					}
					
					tmpa = dum2 * dx + gs[0];
					tmp[0] = dum * tmpa / tmpb;
					tmpa = dum2 * dy + gs[1];
					tmp[1] = dum * tmpa / tmpb;
					tmpa = dum2 * dz + gs[2];
					tmp[2] = dum * tmpa / tmpb;
					if (damp) {
						tmp[0] = tmp[0] * dampval + ene_old * grd_damp * dx / r;
						tmp[1] = tmp[1] * dampval + ene_old * grd_damp * dy / r;
						tmp[2] = tmp[2] * dampval + ene_old * grd_damp * dz / r;
					}
					g[iat][0] = g[iat][0] + p[0] * tmp[0] * emiss[iz[iat]];
					g[iat][1] = g[iat][1] + p[0] * tmp[1] * emiss[iz[iat]];
					g[iat][2] = g[iat][2] + p[0] * tmp[2] * emiss[iz[iat]];
					if (va < 0.5) {
						continue;
					}
					
					tmpb = dum22 * sqrt(va);
					tmpa = -dum2 * dx - gs[0];
					tmp[0] = dum * tmpa / tmpb;
            	    tmpa = -dum2 * dx - gs[1];
            	    tmp[1] = dum * tmpa / tmpb;
            	    tmpa = -dum2 * dx - gs[2];
            	    tmp[2] = dum * tmpa / tmpb;
					if (damp) {
						ene_old_den = sqrt(va * sab);
						ene_old = ene_old_num / ene_old_den;
						tmp[0] = tmp[0] * dampval - ene_old * grd_damp * dx / r;
            	        tmp[1] = tmp[1] * dampval - ene_old * grd_damp * dy / r;
            	        tmp[2] = tmp[2] * dampval - ene_old * grd_damp * dz / r;
					}
					g[iat][0] = g[iat][0] - p[0] * tmp[0] * emiss[iz[iat]];
					g[iat][1] = g[iat][1] - p[0] * tmp[1] * emiss[iz[iat]];
					g[iat][2] = g[iat][2] - p[0] * tmp[2] * emiss[iz[iat]];
				}
*/	
			} //end of j loop
	    	
			if (echo) {
				//printf
			}
			ecp += ea;
		} //end of i loop
	}

	gcp += ecp * p0;

	//printf
	//printf
	//HF-3c special correction
	if (base) {
		//basegrad(n, MAX_ELEM, iz, xyz, 0.7, 0.03, ebas, gbas);
		//TODO : implement basegrad
		//gcp += ebas;
		//printf
		if (grad) {
	//		g = g + gbas;
		}
	}
	return gcp;
}

// short-range bond correction
void srb_egrad(struct Mol * mol, double rscal, double qscal, double * e, double g[][3]) {
	double dx, dy, dz;
	double fi, fj, ff, rf, r;
	double r0ab[MAX_ELEM][MAX_ELEM], r0, thrR, temp, tmpexp;
	int i, j;
	int n = mol->n;
	int * iz = mol->atoms;
	
	//setr0ab(r0ab);

	thrR = 30; //threshold is 30 bohr
	*e = 0;
	for (i = 0; i < n; i++) {
		g[0][i] = 0;
		g[1][i] = 0;
		g[2][i] = 0;
	}

	for (i = 0; i < n - 1; i++) {
		for (j = i + 1; j < n; j++) {
			dx = mol->x[i] - mol->x[j];
			dy = mol->y[i] - mol->y[j];
			dz = mol->z[i] - mol->z[j];
			r = sqrt(dx * dx + dy * dy + dz * dz);
			if (r > thrR) {
				continue;
			}
			r0 = rscal / r0ab[iz[i]][iz[j]];
			fi = iz[i];
			fj = iz[j];
			ff = -sqrt(fi * fj);
			tmpexp = exp(-r0 * r);
			*e += ff * tmpexp;
			rf = qscal / r;
			temp = ff * r0 * dx * tmpexp * rf;
			g[i][0] -= temp;
			g[j][0] += temp;
			temp = ff * r0 * dy * tmpexp * rf;
			g[i][1] -= temp;
			g[j][1] += temp;
			temp = ff * r0 * dz * tmpexp * rf;
			g[i][2] -= temp;
			g[j][2] += temp;
		}
	}
	*e *= qscal;
}

// Calculates the s-type overlap integral for 1s, 2s, 3s slater functions
// za = slater exponent for atom A, zb = slater exponent for atom B
double ssovl(double r, int iat, int jat, int * iz, double za, double zb) {
	int i, ii, shell[73];
	double R, ovl, ax, bx, norm, R05, xx;
	int na, nb;

	na = iz[iat];
	nb = iz[jat];
	
	shell[1] = shell[2] = 1; // h, he
	shell[3] = shell[4] = shell[5] = shell[6] = shell[7] = shell[8] = shell[9] = shell[10] = 2; // li-ne
	for (i = 10; i < 73; i++) {
		shell[i] = 3; //na-rn, 4s, 5s treated as 3s
	}
	// no f elements
	
	ii = shell[na] * shell[nb];
	// ii determines the kind of ovl:
	// kind		<1s|1s>  <2s|1s>  <2s|2s>  <1s|3s>  <2s|3s>  <3s|3s>
	// case:       1        2        4        3        6        9
	R05 = r * 0.5;
	if (shell[na] < shell[nb]) {
		ax = (za + zb) * R05;
		bx = (zb - za) * R05;
	}
	else {
		// swap za and zb to calculate <ns|ms> with n > m
		xx = za;
		za = zb;
		zb = xx;
		ax = (za + zb) * R05;
		bx = (zb - za) * R05;
	}
	if (fabs(za - zb) < 0.1) {
		switch (ii) {
			case 1: // <1s|1s>
				return 0.25 * sqrt(pow(za * zb * r * r,3)) * (a2(ax) * bint(bx, 0) - bint(bx, 2) * a0(ax));
			case 2: // <1s|2s> or <2s|1s>
				ovl = sqrt(1.0 / 3.0);
				norm = sqrt(pow(za, 3) * pow(zb, 5)) * pow(r, 4) * 0.125;
				return ovl * norm * (a3(ax) * bint(bx, 0) - bint(bx, 3) * a0(ax) + a2(ax) * bint(bx, 1) - bint(bx, 2) * a1(ax));
			case 4: // <2s|2s>
				norm = sqrt(pow(za * zb, 5)) * pow(r, 5) * 0.0625;
				return norm * (a4(ax) * bint(bx, 0) + bint(bx, 4) * a0(ax) - 2 * a2(ax) * bint(bx, 2)) / 3.0;
			case 3: // <1s|3s> or <3s|1s>
				norm = sqrt(pow(za, 3) * pow(zb, 7) / 7.5) * pow(r, 5) * 0.0625;
				return norm * (a4(ax) * bint(bx, 0) - a0(ax) * bint(bx, 4) + 2 * (a3(ax) * bint(bx, 1) - a1(ax) * bint(bx, 3))) / sqrt(3.0);
			case 6: // <2s|3s> or <3s|2s>
				norm = sqrt(pow(za * zb, 5) * zb * zb / 7.5) * pow(r, 6) * 0.03125;
				return norm * (a5(ax) * bint(bx, 0) + a4(ax) * bint(bx, 1) - 2 * (a3(ax) * bint(bx, 2) + a2(ax) * bint(bx, 3)) + a1(ax) * bint(bx, 4) + a0(ax) * bint(bx, 5)) / 3.0;
			case 9: // <3s|3s>
				norm = sqrt(pow(za * zb * r * r, 7)) / 480.0;
				return norm * (a6(ax) * bint(bx, 0) - 3 * (a4(ax) * bint(bx, 2) - a2(ax) * bint(bx, 4)) - a0(ax) * bint(bx, 6)) / 3.0;
		}
	}
	else { // different elements
		switch(ii) {
			case 1: // <1s|1s>
				return 0.25 * sqrt(pow(za * zb * r * r,3)) * (a2(ax) * b0(bx) - b2(bx) * a0(ax));
			case 2: // <1s|2s> or <2s|1s>
				norm = sqrt(pow(za * zb,3) * zb * zb) * pow(r, 4) * 0.125;
				return norm * (a3(ax) * b0(bx) - b3(bx) * a0(ax) + a2(ax) * b1(bx) - b2(bx) * a1(ax)) / sqrt(3);
			case 4: // <2s|2s>
				norm = sqrt(pow(za * zb, 5)) * pow(r, 5) * 0.0625;
				return norm * (a4(ax) * b0(bx) + a0(ax) * b4(bx) - 2 * a2(ax) * b2(bx)) / 3.0;
			case 3: // <1s|3s> or <3s|1s>
				norm = sqrt(pow(za * zb, 3) * pow(zb, 4) / 7.5) * pow(r, 5) * 0.0625;
				return norm * (a4(ax) * b0(bx) - b4(bx) * a0(ax) + 2 * (a3(ax) * b1(bx) - b3(bx) * a1(ax))) / sqrt(3);
			case 6: // <2s|3s> or <3s|2s>
				norm = sqrt(pow(za * zb, 5) * zb * zb / 7.5) * pow(r, 6) * 0.03125;
				return norm * (a5(ax) * b0(bx) +a4(ax) * b1(bx) - 2 * (a3(ax) * b2(bx) + a2(ax) * b3(bx)) + a1(ax) * b4(bx) + a0(ax) * b5(bx)) / 3.0;
			case 9: // <3s|3s>
				norm = sqrt(pow(za * zb * r * r, 7)) / 1440.0;
				return norm * (a6(ax) * b0(bx) - 3 * (a4(ax) * b2(bx) - a2(ax) * b4(bx)) - a0(ax) * b6(bx));
			}
		}
	return -1;
}

// a(x) auxiliary integrals

double a0(double x) {
	return exp(-x) / x;
}

double a1(double x) {
	return ((1 + x) * exp(-x)) / (x * x);
}

double a2(double x) {
	double x2 = x * x;
	double x3 = x2 * x;
	return (2 + 2 * x + x2) * exp(-x) / x3;
}

double a3(double x) {
	double x2 = x * x;
	double x3 = x2 * x;
	double x4 = x3 * x;
	return (6 + 6 * x + 3 * x2 + x3) * exp(-x) / x4;
}

double a4(double x) {
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
	double x5 = x4 * x;
	return (24 + 24 * x + 12 * x2 + 4 * x3 + x4) * exp(-x) / x5;
}

double a5(double x) {
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
    double x5 = x4 * x;
	double x6 = x5 * x;
	return (120 + 120 * x + 60 * x2 + 20 * x3 + 5 * x4 + x5) * exp(-x) / x6;
}

double a6(double x) {
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
    double x5 = x4 * x;
    double x6 = x5 * x;
	double x7 = x6 * x;
	return (720 + 720 * x + 360 * x2 + 120 * x3 + 30 * x4 + 6 * x5 + x6) * exp(-x) / x7;
}

// b(x) auxiliary integrals

double b0(double x) {
	return (exp(x) - exp(-x)) / x;
}

double b1(double x) {
	double x2 = x * x;
	return ((1 - x) * exp(x) - (1 + x) * exp(-x)) / x2;
}

double b2(double x) {
	double x2 = x * x;
	double x3 = x2 * x;
	return (((2 - 2 * x + x2) * exp(x)) - ((2 + 2 * x + x2) * exp(-x))) / x3;
}

double b3(double x) {
	double x2 = x * x;
	double x3 = x2 * x;
	double x4 = x3 * x;
	return ((6 - 6 * x + 3 * x2 - x3) * exp(x) - (6 + 6 * x + 3 * x2 + x3) * exp(-x)) / x4;
}

double b4(double x) {
	double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
	double x5 = x4 * x;
	return ((24 - 24 * x + 12 * x2 - 4 * x3 + x4) * exp(x) - (24 + 24 * x + 12 * x2 + 4 * x3 + x4) * exp(-x)) / x5;
}

double b5(double x) {
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
    double x5 = x4 * x;
	double x6 = x5 * x;
	return ((120 - 120 * x + 60 * x2 - 20 * x3 + 5 * x4 - x5) * exp(x) - (120 + 120 * x + 60 * x2 + 20 * x3 + 5 * x4 + x5) * exp(-x)) / x6;
}

double b6(double x) {
	double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
    double x5 = x4 * x;
    double x6 = x5 * x;
	double x7 = x6 * x;
	return ((720 - 720 * x + 360 * x2 - 120 * x3 + 30 * x4 - 6 * x5 + x6) * exp(x) - (720 + 720 * x + 360 * x2 + 120 * x3 + 30 * x4 + 6 * x5 + x6) * exp(-x)) / x7;
}

// calculates B_k(x)
// general summation formula
// 12 terms seems accurate enough for numerical sum
double bint(double x, int k) {
	int i;
	double bint = 0;
	
	if (fabs(x) < 0.000001) {
		bint = (1 + pow(-1, k)) / (k + 1.0);
		return bint;
	}
	
	
	for (i = 0; i < 13; i++) {
		bint += (1 - pow(-1, k + i + 1)) / (fact(i) * (k + i + 1)) * pow(-x, i);
	}
	return bint;
}

int fact(int i) {
	if (i > 0) {
		int k = i;
		while (k > 1) {
			k--;
			i *= k;
		}
		return i;
	}
	return 1;
}

// Calculates the gradient of the s-type overlap for 1s, 2s, 3s slater functions
double gsovl(double r, int iat, int jat, int * iz, double za, double zb) {

	int i, ii, shell[73], same;
	double ovl, ax, bx, norm, R05, xx;
	int na, nb;

	na = iz[iat];
	nb = iz[jat];
	
	shell[1] = shell[2] = 1; // h, he
	shell[3] = shell[4] = shell[5] = shell[6] = shell[7] = shell[8] = shell[9] = shell[10] = 2; // li-ne
	for (i = 10; i < 73; i++) {
		shell[i] = 3; //na-rn, 4s, 5s treated as 3s
	}
	// no f elements
	
	ii = shell[na] * shell[nb];
	// ii determines the kind of ovl:
	// kind		<1s|1s>  <2s|1s>  <2s|2s>  <1s|3s>  <2s|3s>  <3s|3s>
	// case:       1        2        4        3        6        9
	R05 = r * 0.5;
	double fa = za + zb;
	double fb = zb - za;
	ax = fa * R05;
	bx = fb * R05;

	same = (fabs(za - zb) < 0.1) ? 0 : 1;
	switch(ii) {
		case 1:
			return g1s1s(za, zb, fa, fb, r, same);
		case 2:
			if (shell[na] < shell[nb]) {
				return g2s1s(za, zb, fa, fb, r, same);
			}
			else {
				xx = za;	
				za = zb;
				zb = xx;
				return g2s1s(za, zb, fa, -fb, r, same);
			}
		case 4:
			return g2s2s(za, zb, fa, fb, r, same);
		case 3:
			if (shell[na] < shell[nb]) {
				return g3s1s(za, zb, fa, fb, r, same);
			}
			else {
				xx = za;
				za = zb; 
				zb = xx;
				return g3s1s(za, zb, fa, -fb, r, same);
			}
		case 6:
			if (shell[na] < shell[nb]) {
				return g3s2s(za, zb, fa, fb, r, same);
			}
			else {
				xx = za;
				za = zb; 
				zb = xx;
				return g3s2s(za, zb, fa, -fb, r, same);
			}
		case 9:
			return g3s3s(za, zb, fa, fb, r, same);
	}
	return -1;		
}

void setzet(double eta, double * za) {
	int i; 
	double zs[36] = {1.2000,1.6469,0.6534,1.0365,1.3990,1.7210,2.0348,2.2399,2.5644,2.8812,0.8675,1.1935,1.5143,1.7580,1.9860,2.1362,2.3617,2.5796,0.9362,1.2112,1.2870,1.3416,1.3570,1.3804,1.4761,1.5465,1.5650,1.5532,1.5781,1.7778,2.0675,2.2702,2.4546,2.5680,2.7523,2.9299};
	double zp[36] = {0.0000,0.0000,0.5305,0.8994,1.2685,1.6105,1.9398,2.0477,2.4022,2.7421,0.6148,0.8809,1.1660,1.4337,1.6755,1.7721,2.0176,2.2501,0.6914,0.9329,0.9828,1.0104,0.9947,0.9784,1.0641,1.1114,1.1001,1.0594,1.0527,1.2448,1.5073,1.7680,1.9819,2.0548,2.2652,2.4617};
	double zd[36] = {0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,2.4341,2.6439,2.7809,2.9775,3.2208,3.4537,3.6023,3.7017,3.8962,2.0477,2.4022,2.7421,0.6148,0.8809,1.1660,1.4337};

//	printf("passed eta %f\n",eta);
	for (i = 1; i < 37; i++) {
		if (i < 3) {
			za[i] = zs[i-1] * eta;
			//printf("setting za[%d] = %f\n",i,za[i]);
		}
		else if (i > 20 && i < 31) {
			za[i] = (zs[i-1] + zp[i-1] + zd[i-1]) / 3.0 * eta;
			//printf("setting za[%d] = %f\n",i,za[i]);
		}
		else {
			za[i] = (zs[i-1] + zp[i-1]) / 2.0 * eta;
			//printf("setting za[%d] = %f\n",i,za[i]);
		}
	}
}

//-------------------------------------------------------------
// Maple was used to find the analy. derivatives of
// the slater integrals (expressions over A,B aux. integrals) 
// Optimized fortran code by maple with some human-corrections
// And then translated to C by a script
// (as if it couldn't get any more unreadable)
//-------------------------------------------------------------
double g1s1s(double za,double zb,double fa,double fb,double r,int sameEl) {
// slater overlap derv.
// derivative of explicit integral expression
// using maple 

	if(sameEl)  {
		double t1=za * za;
		double t3=zb * zb;
		double   t5 = t1 * za * t3 * zb;
		double t6=r * r;
		double t7=t6 * t6;
		double   t10 = fa * r;
		double   t14 = exp(-0.5E0 * t10);
		double   t17 = sqrt(t5 * t7 * t6);
		double   g = -(1/3) * t5 * t7 / fa * (0.2E1 + t10) * t14 / t17;
		return g;
	}
	else {
		double t1=za * za;
		double t3=zb * zb;
		double   t5 = t1 * za * t3 * zb;
		double t6=fb * fb;
		double   t7 = fb * r;
		double   t8 = 0.5E0 * t7;
		double   t9 = exp(t8);
		double   t12 = exp(-t8);
		double   t15 = t6 * fa;
		double t22=fa * fa;
		double   t23 = t22 * t9;
		double   t27 = t22 * t12;
		double   t31 = t6 * fb;
		double   t32 = r * t31;
		double   t37 = t22 * fa;
		double   t38 = r * t37;
		double t43=r * r;
		double   t44 = t43 * t31;
		double   t51 = t43 * t37;
		double   t56 = 0.4E1 * t6 * t9 - 0.4E1 * t6 * t12 + 0.2E1 * t15 * r * t9 -            0.2E1 * t15 * r * t12 - 0.4E1 * t23 + 0.2E1 * t23 * t7 + 0.4E1 * t27         + 0.2E1 * t27 * t7 - 0.2E1 * t32 * t9 - 0.2E1 * t32 * t12 - 0.2E1 * t38 *    t9 + 0.2E1 * t38 * t12 - 0.1E1 * t44 * fa * t9 - 0.1E1                       * t44 * fa * t12 + t51 * t9 * fb + t51 * t12 * fb;
		double   t61 = exp(-0.5E0 * fa * r);
		double t62=t43 * t43;
		double   t65 = sqrt(t5 * t62 * t43);
		double   g = -0.2E1 * t5 * r * t56 * t61 / t65 / t31 / t37;
		return g;
	}


	return -1;
}


double g2s1s(double za,double zb,double fa,double fb,double r,int sameEl) {
// slater overlap derv.
// derivative of explicit integral expression
// using maple 
		double norm=(1/24)*sqrt(pow(za * zb,3)*zb * 3*zb);

	if(sameEl)  {

		double       t1 = fa * r;
		double       t3 = exp(-0.5000000000E0 * t1);
		double t6=fa * fa;
		double t7=r * r;
		double       g = -0.1000000000E-8 * r * t3 * (0.5333333380E10 + 0.2666666670E10       * t1 + 0.1333333333E10 * t6 * t7) / t6;
		g=g*norm;
		return g;
	}
	else {

		double       t3 = exp(-0.5000000000E0 * fa * r);
		double t4=fa * fa;
		double       t5 = t4 * fa;
		double       t6 = fb * r;
		double       t7 = 0.5000000000E0 * t6;
		double       t8 = exp(t7);
		double       t9 = t5 * t8;
		double t11=fb * fb;
		double       t12 = t11 * fa;
		double       t15 = exp(-t7);
		double t18=t4 * t4;
		double       t19 = r * t18;
		double t22=t11 * t11;
		double       t29 = fb * t4;
		double t36=r * r;
		double       t37 = t36 * t18;
		double       t44 = t36 * r;
		double       t48 = -0.12E2 * t9 + 0.4E1 * t12 * t8 - 0.4E1 * t12 * t15       - 0.6E1 * t19 * t8 - 0.6E1 * t22 * t8 * r - 0.6E1 * t22 * t15       * r + 0.4E1 * t29 * t15 - 0.4E1 * t29 * t8 + 0.6E1 * t19 * t15       + 0.2E1 * t37 * t8 * fb + 0.4E1 * t37 * t15 * fb + t44 * t18 * t15 * t11;
		double       t49 = t5 * t15;
		double       t51 = t11 * fb;
		double       t58 = t51 * fa;
		double       t59 = r * t8;
		double       t76 = t36 * t15;
		double       t79 = t22 * fa;
		double       t87 = 0.12E2 * t49 - 0.12E2 * t51 * t15 - 0.1E1 * t22 * t4 * t15 * t44       + 0.4E1 * t58 * t59 - 0.8E1 * t58 * r * t15 + 0.4E1 * t9 * t6 + 0.8E1 *       t49 * t6 + 0.2E1 * t49 * t11 * t36 + 0.4E1 * t11 * t4 * t59 - 0.2E1 * t51       * t4 * t76 - 0.2E1 * t79 * t36 * t8 - 0.4E1 * t79 * t76 + 0.12E2 * t51 * t8;
		double       g = -0.16E2 * t3 * (t48 + t87) / t36 / t22 / t18;
		g=g*norm;

		return g;
	}
	return -1;
}

double g2s2s(double za,double zb,double fa,double fb,double r,int sameEl) {
// slater overlap derv.
// derivative of explicit integral expression
// using maple 

		double norm=1/(48)*sqrt(pow(za*zb,5) * 5);

	if(sameEl)  {

		double t2=r * r;
		double t5=fa * fa;
		double       t9 = t5 * fa;
		double t10=t2 * t2;
		double       t16 = exp(-fa * r / 0.2E1);
		double       g = (-0.4266666666E2 * r - 0.2133333333E2 * fa * t2 - 0.2133333333E1           * t5 * t2 * r - 0.1066666666E1 * t9 * t10) * t16 / t9;
		       g=g*norm;


		return g;
	}
	else {
		double t1=r * r;
		double       t3 = 0.3840000000E3 * t1 * fb;
		double       t4 = t1 * r;
		double t5=fb * fb;
		double       t7 = 0.6400000000E2 * t4 * t5;
		double       t8 = 0.7680000000E3 * r;
		double t10=fa * fa;
		double t11=t10 * t10;
		double       t12 = t11 * fa;
		double       t14 = fb * r;
		double       t15 = 0.768000000E3 * t14;
		double       t17 = 0.1280000000E3 * t5 * t1;
		double       t21 = 0.256000000E3 * t5 * r;
		double       t22 = t5 * fb;
		double       t24 = 0.1280000000E3 * t22 * t1;
		double       t26 = t10 * fa;
		double t28=t5 * t5;
		double       t30 = 0.1280000000E3 * t1 * t28;
		double       t32 = 0.256000000E3 * t22 * r;
		double       t33 = 0.512000000E3 * t5;
		double       t34 = t28 * fb;
		double       t36 = 0.6400000000E2 * t4 * t34;
		double       t40 = 0.768000000E3 * t28 * r;
		double       t42 = 0.3840000000E3 * t1 * t34;
		double       t45 = 0.1536000000E4 * t28;
		double       t47 = 0.7680000000E3 * t34 * r;
		double       t51 = exp(-0.5E0 * fa * r);
		double       t53 = 0.5E0 * t14;
		double       t54 = exp(-t53);
		double       t68 = exp(t53);
		double       g = (((t3 + t7 + t8) * t12 + (0.1536000000E4 + t15 + t17) * t11 +       (-t21 - t24) * t26 + (t30 - t32 - t33 + t36) * t10 + (t40 + t42) *      fa + t45 + t47) * t51 * t54 + ((t3 - t8 - t7) * t12 + (-0.1536000000E4       + t15 - t17) * t11 + (-t24 + t21) * t26 + (-t30 + t33 - t32       + t36) * t10 + (-t40 + t42) * fa + t47 - t45) * t51 * t68) / t1 /        t12 / t34;


		      g=g*norm;


		return g;
	}

	return -1;
}


double g3s1s(double za,double zb,double fa,double fb,double r,int sameEl) {
// slater overlap derv.
// derivative of explicit integral expression
// using maple

		double norm=sqrt(pow(za * zb,3) * pow(zb,4)/7.5)/(16 * sqrt(3));

	if(sameEl)  {

		double   t1 = fa * r;
		double   t3 = exp(-0.5000000000E0 * t1);
		double t4=r * r;
		double   g = -0.1600000000E1 * t3 * t4 * r * (0.2E1 + t1) / fa;
		 g=g*norm;
		return g;
	}
	else {

		double       t3 = exp(-0.5000E0 * fa * r);
		double t4=fb * fb;
		double t5=t4 * t4;
		double       t6 = t5 * fb;
		double       t7 = t6 * fa;
		double t8=r * r;
		double       t9 = fb * r;
		double       t10 = 0.50E0 * t9;
		double       t11 = exp(t10);
		double       t15 = exp(-t10);
		double       t16 = t8 * t15;
		double t19=fa * fa;
		double       t21 = t8 * r;
		double       t22 = t21 * t15;
		double       t25 = t19 * fa;
		double t27=t8 * t8;
		double t31=t19 * t19;
		double       t32 = t31 * fa;
		double       t33 = t8 * t32;
		double       t45 = t4 * fb;
		double       t48 = t31 * t15;
		double       t55 = t4 * t25;
		double       t56 = t11 * r;
		double       t59 = t15 * r;
		double       t62 = t5 * fa;
		double       t73 = -0.6E1 * t7 * t8 * t11 - 0.18E2 * t7 * t16 - 0.6E1 * t6 * t19       * t22 - 0.1E1 * t6 * t25 * t27 * t15 + 0.6E1 * t33 * t11 * fb + 0.18E2       * t33 * t15 * fb + 0.6E1 * t21 * t32 * t15 * t4 + t27 * t32* t15 * t45       + 0.2E1 * t48 * t45 * t21 + 0.12E2 * t48 * t4 * t8 + 0.12E2 * t55 * t56       + 0.12E2 * t55 * t59 + 0.12E2 * t62 * t56 - 0.36E2 * t62 * t59 - 0.12E2       * t5 * t19 * t16 - 0.2E1 * t5 * t25 * t22;
		double       t74 = t31 * t11;
		double       t79 = t45 * t19;
		double       t92 = r * t32;
		double       t95 = t45 * fa;
		double       t100 = fb * t25;
		double       t111 = 0.12E2 * t74 * t9 + 0.36E2 * t48 * t9 + 0.12E2 * t79 * t56 - 0.12E2        * t79 * t59 + 0.48E2 * t5 * t11 - 0.24E2 * t6 * t11 * r - 0.24E2 * t6 * t15       * r - 0.24E2 * t92 * t11 + 0.24E2 * t95 * t11 - 0.24E2 * t95 * t15 + 0.24E2       * t100 * t15 + 0.24E2 * t92 * t15 - 0.24E2 * t100 * t11 - 0.48E2 * t5 * t15       - 0.48E2 * t74 + 0.48E2 * t48;
		double       g = -0.32E2 * t3 * (t73 + t111) / t8 / t6 / t32;

		   g=g*norm;

		return g;
	}
	return -1;
}


double g3s2s(double za,double zb,double fa,double fb,double r,int sameEl) {
// slater overlap derv.
// derivative of explicit integral expression
// using maple
		double norm=sqrt(pow(za*zb,5)*zb * zb / 7.5)/96;

	if(sameEl)  {
		double       t1 = fa * r;
		double       t3 = exp(-0.5000000000E0 * t1);
		double t6=fa * fa;
		double t7=r * r;
		double t14=t6 * t6;
		double t15=t7 * t7;
		double       g = -0.2000000000E-8 * r * t3 * (0.1280000000E12 + 0.6400000000E11       * t1 + 0.1280000000E11 * t6 * t7 + 0.1066666670E10 * t6 * fa * t7       * r + 0.533333333E9 * t14 * t15) / t14;
		     g=g*norm;
	}
	else {

		double       t3 = exp(-0.5E0 * fa * r);
		double t4=fb * fb;
		double t5=t4 * t4;
		double t6=fa * fa;
		double       t7 = t6 * fa;
		double       t8 = t5 * t7;
		double t9=r * r;
		double       t11 = 0.50E0 * fb * r;
		double       t12 = exp(t11);
		double       t13 = t9 * t12;
		double t16=t6 * t6;
		double       t17 = t16 * fa;
		double       t18 = exp(-t11);
		double       t21 = t5 * fb;
		double       t28 = t9 * t18;
		double       t32 = t9 * r;
		double       t33 = t32 * t18;
		double       t36 = t5 * t4;
		double t38=t9 * t9;
		double       t39 = t38 * t18;
		double       t41 = t21 * fa;
		double       t42 = r * t12;
		double       t45 = t16 * t6;
		double       t46 = t4 * fb;
		double       t49 = t46 * t16;
		double       t52 = -0.6E1 * t8 * t13 + 0.120E3 * t17 * t18 + 0.120E3 * t21 * t18        - 0.120E3 * t17 * t12 - 0.120E3 * t21 * t12 - 0.6E1 * t8 * t28 - 0.2E1       * t5 * t16 * t33 + t36 * t7 * t39 - 0.48E2 * t41 * t42 + t45 * t46 * t39 - 0.6E1 * t49 * t13;
		double       t54 = r * t18;
		double       t60 = t46 * t6;
		double       t63 = fb * t16;
		double       t66 = t5 * fa;
		double       t69 = t4 * t7;
		double       t72 = t36 * t6;
		double       t75 = t32 * t12;
		double       t78 = fb * t9;
		double       t84 = fb * t17;
		double       t87 = -0.24E2 * t46 * t7 * t54 - 0.24E2 * t5 * t6 * t42 + 0.24E2 *       t60 * t12 + 0.24E2 * t63 * t18 - 0.24E2 * t66 * t12 - 0.24E2 * t69       * t18 + 0.9E1 * t72 * t33 + 0.3E1 * t72 * t75 + 0.24E2 * t78 * t45       * t12 - 0.6E1 * t49 * t28 + 0.48E2 * t84 * t42;
		double       t102 = t21 * t6;
		double       t105 = t4 * t17;
		double       t113 = t45 * t4;
		double       t118 = 0.72E2 * t84 * t54 + 0.72E2 * t41 * t54 + 0.36E2 * t78 * t45       * t18 + 0.2E1 * t46 * t17 * t33 + 0.24E2 * t4 * t16 * t42 - 0.6E1         * t102 * t13 - 0.6E1 * t105 * t13 + 0.18E2 * t105 * t28 + 0.2E1       * t21 * t7 * t33 - 0.3E1 * t113 * t75 + 0.9E1 * t113 * t33;
		double       t121 = t36 * fa;
		double       t130 = r * t45;
		double       t145 = 0.18E2 * t102 * t28 + 0.24E2 * t121 * t13 + 0.36E2 * t121 *        t28 - 0.24E2 * t60 * t18 - 0.24E2 * t63 * t12 + 0.60E2 * t130 * t18       + 0.60E2 * t36 * t18 * r + 0.24E2 * t69 * t12 + 0.60E2 * t36 * t12 *       r - 0.60E2 * t130 * t12 + 0.24E2 * t66 * t18;
		double       g = 0.128E3 * t3 * (t52 + t87 + t118 + t145) / t9 / t36 / t45;

		 g=g*norm;

		return g;
	}
	return -1;
}


double g3s3s(double za,double zb,double fa,double fb,double r,int sameEl) {
// slater overlap derv.
// derivative of explicit integral expression
// using maple 

		double norm=sqrt(pow(za * zb,7))/1440;

	if(sameEl)  {

		double       t1 = fa * r;
		double       t3 = exp(-0.5000000000E0 * t1);
		double t5=fa * fa;
		double t6=t5 * t5;
		double       t7 = t6 * fa;
		double t8=r * r;
		double t9=t8 * t8;
		double       g = -0.2000000000E-8 * t3 * r * (0.457142857E9 * t7 * t9       * r + 0.7680000000E12 * t1 + 0.1536000000E12 * t5 * t8       + 0.1280000000E11 * t5 * fa * t8 * r + 0.914285715E9 * t6 * t9 + 0.1536000000E13) / t7;


		  g=g*norm;
		return g;
	}
	else {


		double       t3 = exp(-0.5000000000E0 * fa * r);
		double t4=fa * fa;
		double t5=t4 * t4;
		double       t6 = t5 * t4;
		double       t7 = fb * r;
		double       t8 = 0.5000000000E0 * t7;
		double       t9 = exp(-t8);
		double       t10 = t6 * t9;
		double t13=fb * fb;
		double       t14 = t13 * fb;
		double t15=t13 * t13;
		double       t16 = t15 * t14;
		double t17=r * r;
		double       t18 = t17 * r;
		double       t19 = t16 * t18;
		double       t23 = exp(t8);
		double       t24 = t6 * t23;
		double       t27 = t5 * fa;
		double       t28 = t27 * t13;
		double       t29 = r * t23;
		double       t32 = t6 * t13;
		double       t33 = t17 * t9;
		double       t36 = t15 * fb;
		double       t37 = t4 * t36;
		double       t38 = t9 * r;
		double       t43 = t17 * t23;
		double       t46 = t4 * fa;
		double       t47 = t5 * t46;
		double       t48 = t47 * t18;
		double       t52 = t47 * t17;
		double       t65 = 0.120E3 * t10 * t7 - 0.12E2 * t19 * t4 * t9 + 0.120E3       * t24 * t7 + 0.24E2 * t28 * t29 + 0.24E2 * t32 * t33 + 0.24E2 * t37       * t38 - 0.24E2 * t28 * t38 - 0.24E2 * t32 * t43 - 0.12E2 * t48 * t13       * t23 + 0.60E2 * t52 * t23 * fb + 0.12E2 * t48 * t13 * t9 + 0.60E2       * t52 * t9 * fb - 0.12E2 * t19 * t4 * t23;
		double t66=t17 * t17;
		double       t67 = t16 * t66;
		double       t74 = t27 * t14;
		double       t77 = t6 * t14;
		double       t78 = t18 * t23;
		double       t81 = t46 * t15;
		double       t86 = t27 * t15;
		double       t89 = t5 * t36;
		double       t90 = t18 * t9;
		double       t97 = t46 * t36;
		double       t104 = -0.1E1 * t67 * t46 * t9 - 0.1E1 * t67 * t46 * t23 - 0.12E2       * t74 * t43 + 0.2E1 * t77 * t78 - 0.24E2 * t81 * t29 + 0.24E2 * t81       * t38 + 0.2E1 * t86 * t78 + 0.2E1 * t89 * t90 - 0.2E1 * t86 * t90       + 0.24E2 * t37 * t29 + 0.12E2 * t97 * t33 + 0.2E1 * t89 * t78 - 0.12E2 * t74 * t33;
		double       t108 = t5 * t14;
		double       t111 = t15 * t13;
		double       t112 = t111 * t4;
		double       t117 = t111 * t46;
		double       t122 = t111 * fa;
		double       t129 = t4 * t15;
		double       t132 = t47 * r;
		double       t139 = 0.2E1 * t77 * t90 - 0.24E2 * t108 * t38 + 0.24E2 * t112 * t43       - 0.24E2 * t112 * t33 + 0.2E1 * t117 * t78 - 0.2E1 * t117 * t90 + 0.120E3       * t122 * t29 - 0.120E3 * t122 * t38 + 0.12E2 * t97 * t43 - 0.48E2 * t129       * t23 + 0.120E3 * t132 * t9 - 0.120E3 * t132 * t23 + 0.240E3 * t111 * t23;
		double       t140 = t47 * t66;
		double       t145 = t16 * r;
		double       t150 = t16 * t17;
		double       t160 = t5 * t13;
		double       t170 = t140 * t14 * t23 + t140 * t14 * t9 - 0.120E3 * t145 * t9 - 0.24E2       * t108 * t29 - 0.60E2 * t150 * fa * t23 - 0.240E3 * t111 * t9 - 0.240E3       * t24 + 0.240E3 * t10 + 0.48E2 * t129 * t9 - 0.48E2 * t160 * t9 + 0.48E2       * t160 * t23 - 0.120E3 * t145 * t23 - 0.60E2 * t150 * fa * t9;
		double       g = -0.768E3 * t3 * (t65 + t104 + t139 + t170) / t17 / t47 / t16;

		  g=g*norm;
		return g;
	}

	return -1;
}
