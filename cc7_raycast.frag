#version 330 core

#define SCALE_M     (0.00065104166)		// 1/1536
#define SCALE_Tj    (0.00260416666)		// 1/384
#define SCALE_Tij   (0.0625)			// 1/16

in vec2 vTexCoord;

uniform sampler2D tex_back;   // texture for exiting positions of rays
uniform sampler2D tex_front;  // texture for entering positions of rays
uniform sampler3D tex_volume; // texture for volume dataset
uniform vec3 scale_axes; // scaling from lattice space to (normalized) bounding box space
uniform vec3 dim;    // volume dataset resolution
uniform float level;      // isosurface level
uniform mat4 MV;         // model-view matrix

float scale_step = 0.001; 

// multi render targets
layout (location=0) out vec4 fPosition; // position
layout (location=1) out vec4 fGradient; // gradient
layout (location=2) out vec4 fHessian1;  // Hessian (Dxx, Dyy, Dzz)
layout (location=3) out vec4 fHessian2;  // Hessian (Dxy, Dxz, Dyz)

float c[53];
vec4  u;
mat3    P;
ivec3 org;
int[4] vecR;
int[6] vecP;
ivec3 type_R;
int idx_R, idx_P, idx;

#define	EVAL(p)	(preprocess(p), fetch_coefficients(), eval_f())

#define	GET_DATA(texcoords)	texelFetch(tex_volume, texcoords, 0).r

#define	u0	u.x
#define	u1	u.y
#define	u2	u.z
#define	u3	u.w

#define	c0	c[0]
#define	c1	c[1]
#define	c2	c[2]
#define	c3	c[3]
#define	c4	c[4]
#define	c5	c[5]
#define	c6	c[6]
#define	c7	c[7]
#define	c8	c[8]
#define	c9	c[9]
#define	c10	c[10]
#define	c11	c[11]
#define	c12	c[12]
#define	c13	c[13]
#define	c14	c[14]
#define	c15	c[15]
#define	c16	c[16]
#define	c17	c[17]
#define	c18	c[18]
#define	c19	c[19]
#define	c20	c[20]
#define	c21	c[21]
#define	c22	c[22]
#define	c23	c[23]
#define	c24	c[24]
#define	c25	c[25]
#define	c26	c[26]
#define	c27	c[27]
#define	c28	c[28]
#define	c29	c[29]
#define	c30	c[30]
#define	c31	c[31]
#define	c32	c[32]
#define	c33	c[33]
#define	c34	c[34]
#define	c35	c[35]
#define	c36	c[36]
#define	c37	c[37]
#define	c38	c[38]
#define	c39	c[39]
#define	c40	c[40]
#define	c41	c[41]
#define	c42	c[42]
#define	c43	c[43]
#define	c44	c[44]
#define	c45	c[45]
#define	c46	c[46]
#define	c47	c[47]
#define	c48	c[48]
#define	c49	c[49]
#define	c50	c[50]
#define	c51	c[51]
#define	c52	c[52]



void preprocess(vec3 p_in)
{
    org = ivec3(round(p_in));   // local origin lattice point

    vec4 p_local = vec4(p_in - vec3(org), 1);    // local coordinates

    // computes the membership against the six knot planes intersecting the unit cube centered at the local origin
    int[6]	d = int[](  int(p_local.z-p_local.y>=0),
                        int(p_local.z-p_local.x>=0),
                        int(p_local.y-p_local.x>=0),
                        int(p_local.x+p_local.y>=0),
                        int(p_local.x+p_local.z>=0),
                        int(p_local.y+p_local.z>=0));

    // type_R: the `even sign change transformation' type of p_local.
    // The even sign change matrix R for each type:
    // (0,0,0): [ 1, 0, 0] (0,1,1): [ 1, 0, 0] (1,0,1): [-1, 0, 0] (1,1,0): [-1, 0, 0]
    //          [ 0, 1, 0]          [ 0,-1, 0]          [ 0, 1, 0]          [ 0,-1, 0]
    //          [ 0, 0, 1]          [ 0, 0,-1]          [ 0, 0,-1]          [ 0, 0, 1]
    type_R = d[3]*d[4]*d[5]*ivec3(0,0,0)
            + (1-d[1])*(1-d[2])*(1-d[5])*ivec3(0,1,1)
            + (1-d[0])*d[2]*(1-d[4])*ivec3(1,0,1)
            + d[0]*d[1]*(1-d[3])*ivec3(1,1,0);

    // serialize type_R. Each corresponds to X_0, X_6, X_12, X_18 in Table 1, respectively.
    // (0,0,0)--> 0, (0,1,1)--> 1, (1,0,1)--> 2, (1,1,0)--> 3
    idx_R = 2*type_R.x + type_R.y;

    // store type_R in a vector form
    vecR = int[](int(idx_R==0),int(idx_R==1),int(idx_R==2),int(idx_R==3));

    // Transform p_local into the `reference left coset' (Fig 2(a)) by the transformation computed above.
    // Same as R^-1*p_local (R is one of the even sign change matrices defined above)
    // Note that R^{-1}=R since R is symmetric & orthogonal.
    vec3  p_ref_R = p_local.xyz * vec3(1 - 2*type_R);   

    // Compute the membership against the three knot planes intersecting the piece in Fig 2(a).
    // Three knot planes with their normals (-1,1,0), (-1,0,1), and (0,-1,1), respectively.
    ivec3 type_P = ivec3(   int(p_ref_R.y-p_ref_R.x>=0),    
                            int(p_ref_R.z-p_ref_R.x>=0), 
                            int(p_ref_R.z-p_ref_R.y>=0));

    // serialize type_P. Each corresponds to X_0~X_5 in Taqle 1, respectively.
    // (0,0,0)--> 0, (0,0,1)--> 1, (1,0,0)--> 2, (0,1,1)--> 3, (1,1,1)--> 4, (1,1,0)--> 5
    idx_P = (1-type_P.x)*type_P.z + type_P.x*type_P.y*(1-type_P.z) + (type_P.x+type_P.y)*2;

    // store type_P in a vector form
    vecP = int[](int(idx_P==0),int(idx_P==1),int(idx_P==2),int(idx_P==3),int(idx_P==4),int(idx_P==5));

    // Compute the permutation matrix P from type_P.
    // (0,0,0):[1,0,0] (0,0,1):[1,0,0] (1,0,0):[0,1,0] (0,1,1):[0,0,1] (1,1,1):[0,0,1] (1,1,0):[0,1,0]
    //         [0,1,0]         [0,0,1]         [1,0,0]         [1,0,0]         [0,1,0]         [0,0,1]
    //         [0,0,1]         [0,1,0]         [0,0,1]         [0,1,0]         [1,0,0]         [1,0,0]
    // For p_ref_R in one of the six tetrahedral pieces, P^{-1}*p_ref_R is inside the reference tetrahedron.
    // Note that mat3 is in column-major format.
    P = mat3(vecP[0]+vecP[1], vecP[2]+vecP[3], vecP[4]+vecP[5],
                  vecP[2]+vecP[5], vecP[0]+vecP[4], vecP[1]+vecP[3],
                  vecP[3]+vecP[4], vecP[1]+vecP[5], vecP[0]+vecP[2]);

    idx = idx_R*6 + idx_P;  // serialize 24 tetrahedra types corresponding to R*P in Table 1.

    // Transform p_ref_R into the `reference tetrahedron' by multiplying P.
    vec4    p_ref = vec4(P*p_ref_R, 1);

    // Compute the barycentric coordinates.
    //     [-2  0  0  1]                      [-2  0  0  1]   [ 0  1/2 1/2 1/2 ]^(-1)
    // u = [ 2 -2  0  0] * p_ref    Note that [ 2 -2  0  0] = [ 0   0  1/2 1/2 ]
    //     [ 0  1 -1  0]                      [ 0  1 -1  0]   [ 0   0 -1/2 1/2 ]
    //     [ 0  1  1  0]                      [ 0  1  1  0]   [ 1   1   1   1  ]
    u = vec4(p_ref.w - 2.0*p_ref.x,
             2.0*(p_ref.x-p_ref.y),
             p_ref.y-p_ref.z,
             p_ref.y+p_ref.z);

}

    // (0,0,0)--> 0, (0,1,1)--> 1, (1,0,1)--> 2, (1,1,0)--> 3
void fetch_coefficients(void)
{
    // To fetch the 52 coefficient, we need to transform the stencils by (R*P) which is slow.
    // Instead, since the stencils are still axis-aligned after the transformation,
    // we compute the transformed axes first.
    // (x,y,z) = R*P
    ivec3   r = 1-2*type_R;
    ivec3 x = r*ivec3(vecP[0]+vecP[1], vecP[2]+vecP[5], vecP[3]+vecP[4]);
    ivec3 y = r*ivec3(vecP[2]+vecP[3], vecP[0]+vecP[4], vecP[1]+vecP[5]);
    ivec3 z = r*ivec3(vecP[4]+vecP[5], vecP[1]+vecP[3], vecP[0]+vecP[2]);

    // Fetching the 53 coefficients.

    // We re-ordered the 53 coefficient fetches such that
    // the displacement of adjacent fetches are axis-aligned.
    // As a result, we can move to the next coefficient with a low computational overhead.

    ivec3 coords = org;
#define	FETCH_C(idx_c, offset)	coords += (offset); c[idx_c] = GET_DATA(coords);
    FETCH_C(32,-2*z)    // ( 0, 0,-2)    1
    FETCH_C(41,   x)    // ( 1, 0,-2)    2
    FETCH_C(51,   y)    // ( 1, 1,-2)    3
    FETCH_C(43,  -x)    // ( 0, 1,-2)    4
    
    FETCH_C( 9,   z)    // ( 0, 1,-1)    5
    FETCH_C(39,   y)    // ( 0, 2,-1)    6
    FETCH_C(49,   x)    // ( 1, 2,-1)    7
    FETCH_C(25,  -y)    // ( 1, 1,-1)    8
    FETCH_C(47,   x)    // ( 2, 1,-1)    9
    FETCH_C(21,-3*x)    // (-1, 1,-1)   10
    FETCH_C(11,  -y)    // (-1, 0,-1)   11
    FETCH_C(19,  -y)    // (-1,-1,-1)   12
    FETCH_C( 7,   x)    // ( 0,-1,-1)   13
    FETCH_C(23,   x)    // ( 1,-1,-1)   14
    FETCH_C(45,   x)    // ( 2,-1,-1)   15
    FETCH_C(35,   y)    // ( 2, 0,-1)   16
    FETCH_C(13,  -x)    // ( 1, 0,-1)   17
    FETCH_C( 6,  -x)    // ( 0, 0,-1)   18
    
    FETCH_C( 0,   z)    // ( 0, 0, 0)   19
    FETCH_C( 1,   x)    // ( 1, 0, 0)   20
    FETCH_C(28,-3*x)    // (-2, 0, 0)   21
    FETCH_C( 2,   x)    // (-1, 0, 0)   22
    FETCH_C(16,   y)    // (-1, 1, 0)   23
    FETCH_C(15,-2*y)    // (-1,-1, 0)   24
    FETCH_C( 4,   x)    // ( 0,-1, 0)   25
    FETCH_C(30,  -y)    // ( 0,-2, 0)   26
    FETCH_C(37,   x)    // ( 1,-2, 0)   27
    FETCH_C(17,   y)    // ( 1,-1, 0)   28
    FETCH_C(33,   x)    // ( 2,-1, 0)   29
    FETCH_C(27,   y)    // ( 2, 0, 0)   30
    FETCH_C(34,   y)    // ( 2, 1, 0)   31
    FETCH_C(18,  -x)    // ( 1, 1, 0)   32
    FETCH_C(38,   y)    // ( 1, 2, 0)   33
    FETCH_C(29,  -x)    // ( 0, 2, 0)   34
    FETCH_C( 3,  -y)    // ( 0, 1, 0)   35
    
    FETCH_C(10,   z)    // ( 0, 1, 1)   36
    FETCH_C(40,   y)    // ( 0, 2, 1)   37
    FETCH_C(50,   x)    // ( 1, 2, 1)   38
    FETCH_C(26,  -y)    // ( 1, 1, 1)   39
    FETCH_C(48,   x)    // ( 2, 1, 1)   40
    FETCH_C(22,-3*x)    // (-1, 1, 1)   41
    FETCH_C(12,  -y)    // (-1, 0, 1)   42
    FETCH_C(20,  -y)    // (-1,-1, 1)   43
    FETCH_C( 8,   x)    // ( 0,-1, 1)   44
    FETCH_C(24,   x)    // ( 1,-1, 1)   45
    FETCH_C(46,   x)    // ( 2,-1, 1)   46
    FETCH_C(36,   y)    // ( 2, 0, 1)   47
    FETCH_C(14,  -x)    // ( 1, 0, 1)   48
    FETCH_C( 5,  -x)    // ( 0, 0, 1)   49
    
    FETCH_C(31,   z)    // ( 0, 0, 2)   50
    FETCH_C(42,   x)    // ( 1, 0, 2)   51
    FETCH_C(52,   y)    // ( 1, 1, 2)   52
    FETCH_C(44,  -x)    // ( 0, 1, 2)   53
#undef	FETCH_C
}

// Evaluate spline f(x)
float eval_f(void)
{
    vec4  u_2 = u*u;
    vec4  u_3 = u_2*u;
    vec4  u_4 = u_2*u_2;
    return
        (
        u_4[0]*(264*c0 + 128*(c1+c2+c3+c4+c5+c6) + 32*(c10+c11+c12+c13+c14+c15+c16+c17+c18+c7+c8+c9) + 12*(c19+c20+c21+c22+c23+c24+c25+c26) + 4*(c27+c28+c29+c30+c31+c32) )
        + u_3[0]*u1*(1056*c0 + 608*c1 + 512*(c3+c4+c5+c6) + 416*c2 + 176*(c13+c14+c17+c18) + 128*(c10+c7+c8+c9) + 80*(c11+c12+c15+c16) + 64*(c23+c24+c25+c26) + 32*(c19+c20+c21+c22+c27) + 16*(c29+c30+c31+c32))
        + u_3[0]*u2*(1056*c0 + 608*(c1+c3+c6) + 416*(c2+c4+c5) + 224*(c13+c18+c9) + 128*(c10+c11+c14+c16+c17+c7) + 96*(c25) + 64*(c21+c23+c26) + 32*(c12+c15+c19+c22+c24+c27+c29+c32+c8))
        + u_3[0]*u3*(1056*c0 + 608*(c1+c3+c5) + 416*(c2+c4+c6) + 224*(c10+c14+c18) + 128*(c12+c13+c16+c17+c8+c9) + 96*(c26) + 64*(c22+c24+c25) + 32*(c11+c15+c20+c21+c23+c27+c29+c31+c7))
        + u_2[0]*u_2[1]*( 1536*c0 + 1056*c1 + 720*(c3+c4+c5+c6) + 480*c2 + 360*(c13+c14+c17+c18) + 192*(c10+c7+c8+c9) + 120*(c23+c24+c25+c26) + 96*c27 + 72*(c11+c12+c15+c16) + 24*(c19+c20+c21+c22+c29+c30+c31+c32))
        + u_2[0]*u1*u2*(3072*c0 + 2112*c1 + 1728*(c3+c6) + 1152*(c4+c5) + 960*c2 + 912*(c13+c18) + 672*c9 + 528*(c14+c17) + 384*(c10+c25+c7) + 240*(c11+c16+c23+c26) + 192*c27 + 96*(c21+c24+c29+c32+c8) + 48*(c12+c15+c19+c22))
        + u_2[0]*u1*u3*( 3072*c0 + 2112*c1 + 1728*(c3+c5) + 1152*(c4+c6) + 960*c2 + 912*(c14+c18) + 672*c10 + 528*(c13+c17) + 384*(c26+c8+c9) + 240*(c12+c16+c24+c25) + 192*c27 + 96*(c22+c23+c29+c31+c7) + 48*(c11+c15+c20+c21))
        + u_2[0]*u_2[2]*( 1440*c0 + 960*(c1+c3+c6) + 576*(c13+c18+c9) + 384*(c2+c4+c5) + 288*c25 + 192*(c10+c11+c14+c16+c17+c7) + 96*(c21+c23+c26+c27+c29+c32))
        + u_2[0]*u2*u3*( 3072*c0 + 2112*(c1+c3) + 1344*(c5+c6) + 1152*c18 + 960*(c2+c4) + 672*(c10+c13+c14+c9) + 384*(c16+c17+c25+c26) + 192*(c27+c29) + 96*(c11+c12+c21+c22+c23+c24+c7+c8))
        + u_2[0]*u_2[3]*(1440*c0 + 960*(c1+c3+c5) + 576*(c10+c14+c18) + 384*(c2+c4+c6) + 288*c26 + 192*(c12+c13+c16+c17+c8+c9) + 96*(c22+c24+c25+c27+c29+c31))
        + u0*u_3[1]*( 960*c0 + 800*c1 + 428*(c3+c4+c5+c6) + 308*(c13+c14+c17+c18) + 240*c2 + 120*(c10+c7+c8+c9) + 112*c27 + 96*(c23+c24+c25+c26) + 28*(c11+c12+c15+c16) + 12*(c29+c30+c31+c32) + 8*(c19+c20+c21+c22) + 4*(c33+c34+c35+c36+c37+c38+c41+c42))
        + u0*u_2[1]*u2*( 2880*c0 + 2400*c1 + 1560*(c3+c6) + 1152*(c13+c18) + 1008*(c4+c5) + 720*c2 + 696*(c14+c17) + 624*c9 + 480*c25 + 360*(c10+c7) + 336*c27 + 288*(c23+c26) + 144*(c11+c16) + 96*(c24+c8) + 72*(c29+c32) + 48*c21 + 24*(c12+c15+c19+c22+c34+c35+c38+c41))
        + u0*u_2[1]*u3*( 2880*c0 + 2400*c1 + 1560*(c3+c5) + 1152*(c14+c18) + 1008*(c4+c6) + 720*c2 + 696*(c13+c17) + 624*c10 + 480*c26 + 360*(c8+c9) + 336*c27 + 288*(c24+c25) + 144*(c12+c16) + 96*(c23+c7) + 72*(c29+c31) + 48*c22 + 24*(c11+c15+c20+c21+c34+c36+c38+c42))
        + u0*u1*u_2[2]*( 2688*c0 + 2208*c1 + 1776*(c3+c6) + 1392*(c13+c18) + 1056*c9 + 768*c25 + 672*(c4+c5) + 576*c2 + 480*(c14+c17) + 336*(c10+c7) + 288*c27 + 240*(c11+c16+c23+c26) + 144*(c29+c32) + 96*c21 + 48*(c34+c35+c38+c41))
        + u0*u1*u2*u3*( 5760*c0 + 4800*c1 + 3840*c3 + 2880*c18 + 2400*(c5+c6) + 1728*(c13+c14) + 1632*c4 + 1440*c2 + 1248*(c10+c9) + 1056*c17 + 960*(c25+c26) + 672*c27 + 480*c16 + 288*c29 + 192*(c23+c24+c7+c8) + 96*(c11+c12+c21+c22+c34+c38))
        + u0*u1*u_2[3]*( 2688*c0 + 2208*c1 + 1776*(c3+c5) + 1392*(c14+c18) + 1056*c10 + 768*c26 + 672*(c4+c6) + 576*c2 + 480*(c13+c17) + 336*(c8+c9) + 288*c27 + 240*(c12+c16+c24+c25) + 144*(c29+c31) + 96*c22 + 48*(c34+c36+c38+c42))
        + u0*u_3[2]*( 768*c0 + 640*(c1+c3+c6) + 512*(c13+c18+c9) + 384*c25 + 128*(c2+c4+c5) + 96*(c10+c11+c14+c16+c17+c7) + 64*(c21+c23+c26+c27+c29+c32) + 32*(c34+c35+c38+c39+c41+c43))
        + u0*u_2[2]*u3*( 2688*c0 + 2208*(c1+c3) + 1728*c18 + 1344*c6 + 1056*(c13+c9) + 768*(c25+c5) + 576*(c10+c14+c2+c4) + 384*(c16+c17+c26) + 288*(c27+c29) + 96*(c11+c21+c23+c34+c38+c7))
        + u0*u2*u_2[3]*( 2688*c0 + 2208*(c1+c3) + 1728*c18 + 1344*c5 + 1056*(c10+c14) + 768*(c26+c6) + 576*(c13+c2+c4+c9) + 384*(c16+c17+c25) + 288*(c27+c29) + 96*(c12+c22+c24+c34+c38+c8))
        + u0*u_3[3]*( 768*c0 + 640*(c1+c3+c5) + 512*(c10+c14+c18) + 384*c26 + 128*(c2+c4+c6) + 96*(c12+c13+c16+c17+c8+c9) + 64*(c22+c24+c25+c27+c29+c31) + 32*(c34+c36+c38+c40+c42+c44))
        + u_4[1]*( 220*(c0+c1) + 92*(c13+c14+c17+c18+c3+c4+c5+c6) + 44*(c2+c27) + 27*(c10+c23+c24+c25+c26+c7+c8+c9) + 4*(c11+c12+c15+c16+c33+c34+c35+c36) + 2*(c29+c30+c31+c32+c37+c38+c41+c42) + (c19+c20+c21+c22+c45+c46+c47+c48))
        + u_3[1]*u2*( 880*(c0+c1) + 452*(c13+c18+c3+c6) + 284*(c14+c17+c4+c5) + 184*(c25+c9) + 176*(c2+c27) + 108*(c10+c23+c26+c7) + 32*(c24+c8) + 28*(c11+c16+c34+c35) + 16*(c29+c32+c38+c41) + 8*(c21+c47) + 4*(c12+c15+c19+c22+c33+c36+c45+c48))
        + u_3[1]*u3*( 880*(c0+c1) + 452*(c14+c18+c3+c5) + 284*(c13+c17+c4+c6) + 184*(c10+c26) + 176*(c2+c27) + 108*(c24+c25+c8+c9) + 32*(c23+c7) + 28*(c12+c16+c34+c36) + 16*(c29+c31+c38+c42) + 8*(c22+c48) + 4*(c11+c15+c20+c21+c33+c35+c46+c47))
        + u_2[1]*u_2[2]*( 1224*(c0+c1) + 792*(c13+c18+c3+c6) + 456*(c25+c9) + 288*(c14+c17+c4+c5) + 216*(c2+c27) + 144*(c10+c23+c26+c7) + 72*(c11+c16+c34+c35) + 48*(c29+c32+c38+c41) + 24*(c21+c47))
        + u_2[1]*u2*u3*( 2640*(c0+c1) + 1680*(c18+c3) + 1032*(c13+c14+c5+c6) + 672*(c17+c4) + 552*(c10+c25+c26+c9) + 528*(c2+c27) + 144*(c16+c34) + 96*(c23+c24+c29+c38+c7+c8) + 24*(c11+c12+c21+c22+c35+c36+c47+c48))
        + u_2[1]*u_2[3]*( 1224*(c0+c1) + 792*(c14+c18+c3+c5) + 456*(c10+c26) + 288*(c13+c17+c4+c6) + 216*(c2+c27) + 144*(c24+c25+c8+c9) + 72*(c12+c16+c34+c36) + 48*(c29+c31+c38+c42) + 24*(c22+c48))
        + u1*u_3[2]*( 704*(c0+c1) + 576*(c13+c18+c3+c6) + 448*(c25+c9) + 112*(c14+c17+c4+c5) + 96*(c2+c27) + 80*(c10+c23+c26+c7) + 64*(c11+c16+c34+c35) + 48*(c29+c32+c38+c41) + 32*(c21+c47) + 16*(c39+c43+c49+c51))
        + u1*u_2[2]*u3*( 2448*(c0+c1) + 1968*(c18+c3) + 1200*(c13+c6) + 912*(c25+c9) + 672*(c14+c5) + 480*(c10+c17+c26+c4) + 432*(c2+c27) + 240*(c16+c34) + 192*(c29+c38) + 96*(c23+c7) + 48*(c11+c21+c35+c47))
        + u1*u2*u_2[3]*( 2448*(c0+c1) + 1968*(c18+c3) + 1200*(c14+c5) + 912*(c10+c26) + 672*(c13+c6) + 480*(c17+c25+c4+c9) + 432*(c2+c27) + 240*(c16+c34) + 192*(c29+c38) + 96*(c24+c8) + 48*(c12+c22+c36+c48))
        + u1*u_3[3]*( 704*(c0+c1) + 576*(c14+c18+c3+c5) + 448*(c10+c26) + 112*(c13+c17+c4+c6) + 96*(c2+c27) + 80*(c24+c25+c8+c9) + 64*(c12+c16+c34+c36) + 48*(c29+c31+c38+c42) + 32*(c22+c48) + 16*(c40+c44+c50+c52))
        + u_4[2]*( 144*(c0+c1+c13+c18+c25+c3+c6+c9) + 16*(c10+c11+c14+c16+c17+c2+c21+c23+c26+c27+c29+c32+c34+c35+c38+c39+c4+c41+c43+c47+c49+c5+c51+c7))
        + u_3[2]*u3*( 704*(c0+c1+c18+c3) + 448*(c13+c25+c6+c9) + 128*(c10+c14+c26+c5) + 96*(c16+c17+c2+c27+c29+c34+c38+c4) + 32*(c11+c21+c23+c35+c39+c47+c49+c7))
        + u_2[2]*u_2[3]*( 1152*(c0+c1+c18+c3) + 384*(c10+c13+c14+c25+c26+c5+c6+c9) + 192*(c16+c17+c2+c27+c29+c34+c38+c4))
        + u2*u_3[3]*( 704*(c0+c1+c18+c3) + 448*(c10+c14+c26+c5) + 128*(c13+c25+c6+c9) + 96*(c16+c17+c2+c27+c29+c34+c38+c4) + 32*(c12+c22+c24+c36+c40+c48+c50+c8))
        + u_4[3]*( 144*(c0+c1+c10+c14+c18+c26+c3+c5) + 16*(c12+c13+c16+c17+c2+c22+c24+c25+c27+c29+c31+c34+c36+c38+c4+c40+c42+c44+c48+c50+c52+c6+c8+c9))
        )*SCALE_M
    ;
}

// Evaluate three 1st derivative formulae.
vec3 eval_gj(void)
{
    vec3 expr;
    vec4 u_2 = u*u;
    vec4 u_3 = u_2*u;

    expr.x = 	// g_5(u)
        -(
        u_3[0]*( 48*(c1-c2) + 24*(-c11-c12+c13+c14-c15-c16+c17+c18) + 8*(-c19-c20-c21-c22+c23+c24+c25+c26+c27-c28))
        + u_2[0]*u1*( + 144*(c1-c2) + 96*(c13+c14+c17+c18) + 48*(-c0-c11-c12-c15-c16+c27-c3-c4-c5-c6) + 24*(-c19-c20-c21-c22+c23+c24+c25+c26))
        + u_2[0]*u2*( + 144*(c1-c2) + 120*(c13+c18) + 72*(-c11+c14-c16+c17) + 48*(-c0-c21+c25+c27-c3-c4-c5-c6) + 24*(-c12-c15-c19-c22+c23+c26) )
        + u_2[0]*u3*( + 144*(c1-c2) + 120*(c14+c18) + 72*(-c12+c13-c16+c17) + 48*(-c0-c22+c26+c27-c3-c4-c5-c6) + 24*(-c11-c15-c20-c21+c24+c25) )
        + u0*u_2[1]*( + 144*c1 - 120*c2 + 102*(c13+c14+c17+c18) -96*c0 + 78*(-c3-c4-c5-c6) + 72*c27 + 30*(-c11-c12-c15-c16) + 24*(c23+c24+c25+c26) + 12*(-c10-c19-c20-c21-c22-c7-c8-c9) + 6*(-c29-c30-c31-c32+c33+c34+c35+c36+c37+c38+c41+c42))
        + u0*u1*u2*( + 288*c1 + 240*(c13+c18-c2) -192*c0 + 168*(c14+c17-c3-c6) + 144*(c27-c4-c5) + 96*(-c11-c16+c25) + 48*(-c21+c23+c26-c9) + 24*(-c10-c12-c15-c19-c22-c29-c32+c34+c35+c38+c41-c7))
        + u0*u1*u3*( + 288*c1 + 240*(c14+c18-c2) -192*c0 + 168*(c13+c17-c3-c5) + 144*(c27-c4-c6) + 96*(-c12-c16+c26) + 48*(-c10-c22+c24+c25) + 24*(-c11-c15-c20-c21-c29-c31+c34+c36+c38+c42-c8-c9))
        + u0*u_2[2]*( + 144*c1 + 120*(c13+c18) + 96*(-c0-c2+c25) + 72*(-c11-c16-c3-c6) + 48*(c14+c17-c21+c27-c4-c5-c9) + 24*(-c10+c23+c26-c29-c32+c34+c35+c38+c41-c7))
        + u0*u2*u3*( + 288*(c1+c18) - 240*c2 + 192*(-c0+c13+c14-c3) + 144*(-c16+c17+c27-c4-c5-c6) + 96*(c25+c26) + 48*(-c10-c11-c12-c21-c22-c29+c34+c38-c9))
        + u0*u_2[3]*( + 144*c1 + 120*(c14+c18) + 96*(-c0-c2+c26) + 72*(-c12-c16-c3-c5) + 48*(-c10+c13+c17-c22+c27-c4-c6) + 24*(c24+c25-c29-c31+c34+c36+c38+c42-c8-c9))
        + u_3[1]*( + 40*(-c0+c1) + 32*(-c2+c27) + 30*(c13+c14+c17+c18-c3-c4-c5-c6) + 6*(-c10-c11-c12-c15-c16+c23+c24+c25+c26+c33+c34+c35+c36-c7-c8-c9) + 2*(-c19-c20-c21-c22-c29-c30-c31-c32+c37+c38+c41+c42+c45+c46+c47+c48))
        + u_2[1]*u2*( + 120*(-c0+c1) + 102*(c13+c18-c3-c6) + 96*(-c2+c27) + 78*(c14+c17-c4-c5) + 36*(c25-c9) + 30*(-c11-c16+c34+c35) + 18*(-c10+c23+c26-c7) + 12*(-c21-c29-c32+c38+c41+c47) + 6*(-c12-c15-c19-c22+c33+c36+c45+c48))
        + u_2[1]*u3*( + 120*(-c0+c1) + 102*(c14+c18-c3-c5) + 96*(-c2+c27) + 78*(c13+c17-c4-c6) + 36*(-c10+c26) + 30*(-c12-c16+c34+c36) + 18*(c24+c25-c8-c9) + 12*(-c22-c29-c31+c38+c42+c48) + 6*(-c11-c15-c20-c21+c33+c35+c46+c47))
        + u1*u_2[2]*( + 120*(c1-c0) + 96*(c13+c18-c3-c6) + 72*(c27-c2+c25-c9) + 48*(c34+c35-c11+c14-c16+c17-c4-c5) + 24*(c26 - c10 - c21 + c23 - c29 - c32 + c38 + c41 + c47 - c7))
        + u1*u2*u3*( + 240*(c18 -c0 + c1 - c3) + 192*(c27 - c2) + 168*(c13 + c14 - c5 - c6) + 144*(c17 - c4) + 96*(c34 - c16) + 72*(c25 + c26 - c10 - c9) + 48*(c38 - c29) + 24*(c35 + c36 - c11 - c12 - c21 - c22 + c47 + c48))
        + u1*u_2[3]*( + 120*(c1 -c0) + 96*(c14 + c18 - c3 - c5) + 72*(c26 + c27 - c10 - c2) + 48*(c13 - c12 - c16 + c17 + c34 + c36 - c4 - c6) + 24*(c38 - c22 + c24 + c25 - c29 - c31 + c42 + c48 - c8 - c9))
        + u_3[2]*( + 32*(c13 -c0 + c1 + c18 + c25 - c3 - c6 - c9) + 16*(c34 + c35 - c11 - c16 - c2 - c21 + c27 + c47) + 8*(c14 - c10 + c17 + c23 + c26 - c29 - c32 + c38 - c39 - c4 + c41 - c43 + c49 - c5 + c51 - c7))
        + u_2[2]*u3*( + 120*(c1 -c0 + c18 - c3) + 72*(c13 - c16 - c2 + c25 + c27 + c34 - c6 - c9) + 48*(c14 - c10 + c17 + c26 - c29 + c38 - c4 - c5) + 24*(c35 - c11 - c21 + c47) )
        + u2*u_2[3]*( + 120*(c1 -c0 + c18 - c3) + 72*(c14 - c16 - c10 - c2 + c26 + c27 + c34 - c5) + 48*(c13 + c17 + c25 - c29 + c38 - c4 - c6 - c9) + 24*(c36 - c12 - c22 + c48))
        + u_3[3]*( + 32*(c1 - c10 -c0 + c14 + c18 + c26 - c3 - c5) + 16*(c27 - c12 - c16 - c2 - c22 + c34 + c36 + c48) + 8*(c13 + c17 + c24 + c25 - c29 - c31 + c38 - c4 - c40 + c42 - c44 + c50 + c52 - c6 - c8 - c9))

        );
    expr.y =	// g_6(u)
        (
        u_3[0]*( + 48*(c4 - c3) + 24*(c7 + c8 - c9 -c10 + c15 - c16 + c17 - c18) + 8*(c19 + c20 - c21 - c22 + c23 + c24 - c25 - c26 - c29 + c30))
        + u_2[0]*u1*( + 144*(c4 - c3) + 96*(c17 - c18) + 72*(c7 + c8 - c9 -c10) + 48*(c15 - c16) + 36*(c23 + c24 - c25 - c26) + 24*(c30 - c29) + 12*(c19 + c20 - c21 - c22) )
        + u_2[0]*u2*( + 144*(c4 - c3) + 120*(-c18 - c9) + 72*(c7 - c10 - c16 + c17) + 48*(c0 + c1 + c2 + c23 - c25 - c29 + c5 + c6) + 24*(c15 + c19 - c21 + c24 - c26 + c8) )
        + u_2[0]*u3*( + 144*(c4 - c3) + 120*(-c10 - c18) + 72*(c8 - c9 - c16 + c17) + 48*(c0 + c1 + c2 + c24 - c26 - c29 + c5 + c6) + 24*(c15 + c20 - c22 + c23 - c25 + c7))
        + u0*u_2[1]*( + 138*(c4 - c3) + 114*(c17 - c18) + 66*(c7 + c8 - c9 -c10) + 48*(c23 + c24 - c25 - c26) + 30*(c15 - c16) + 18*(c30 - c29) + 6*(c19 + c20 - c21 - c22 + c33 - c34 + c37 - c38))
        + u0*u1*u2*( - 288*c3 + 264*(c4 - c18) - 216*c9 + 192*c17 + 144*(c7 - c25) + 120*(c23 - c10) + 96*(c0 + c1 - c16) + 72*(c2 - c26 - c29 + c5 + c6) + 48*(c24 + c8) + 24*(c13 + c14 + c15 + c19 - c21 + c27 - c34 - c38))
        + u0*u1*u3*( - 288*c3 + 264*(c4 - c18) - 216*c10 + 192*c17 + 144*(c8 - c26) + 120*(c24 - c9) + 96*(c0 + c1 - c16) + 72*(c2 - c25 - c29 + c5 + c6) + 48*(c23 + c7) + 24*(c13 + c14 + c15 + c20 - c22 + c27 - c34 - c38))
        + u0*u_2[2]*( - 144*c3 + 120*(-c18 - c9) + 96*(c0 - c25 + c4) + 72*(c1 + c17 + c6 + c7) + 48*(c13 - c10 - c16 + c2 + c23 - c29 + c5) + 24*(c11 + c14 - c21 - c26 + c27 + c32 - c34 - c38 - c39 - c43))
        + u0*u2*u3*( + 288*(-c18 - c3) + 240*c4 + 192*(c0 + c1 - c10 - c9) + 144*(c2 - c16 + c17 - c29 + c5 + c6) + 96*(-c25 - c26) + 48*(c13 + c14 + c23 + c24 + c27 - c34 - c38 + c7 + c8))
        + u0*u_2[3]*( - 144*c3 + 120*(-c10 - c18) + 96*(c0 - c26 + c4) + 72*(c1 + c17 + c5 + c8) + 48*(c14 - c16 + c2 + c24 - c29 + c6 - c9) + 24*(c12 + c13 - c22 - c25 + c27 + c31 - c34 - c38 - c40 - c44))
        + u_3[1]*( + 42*(c17 - c18 - c3 + c4) + 19*(c23 + c24 - c25 - c26 -c10 + c7 + c8 - c9) + 6*(c15 - c16 + c33 - c34) + 4*(c30 - c29 + c37 - c38) + (c19 + c20 - c21 - c22 + c45 + c46 - c47 - c48))
        + u_2[1]*u2*( + 138*(-c18 - c3) + 114*(c17 + c4) + 90*(-c25 - c9) + 66*(c23 + c7) + 48*(c0 + c1 - c10 - c26) + 30*(-c16 - c34) + 24*(c13 + c14 + c2 + c24 + c27 - c29 - c38 + c5 + c6 + c8) + 6*(c15 + c19 - c21 + c33 + c45 - c47))
        + u_2[1]*u3*( + 138*(-c18 - c3) + 114*(c17 + c4) + 90*(-c10 - c26) + 66*(c24 + c8) + 48*(c0 + c1 - c25 - c9) + 30*(-c16 - c34) + 24*(c13 + c14 + c2 + c23 + c27 - c29 - c38 + c5 + c6 + c7) + 6*(c15 + c20 - c22 + c33 + c46 - c48))
        + u1*u_2[2]*( + 132*(-c18 - c3) + 108*(-c25 - c9) + 84*(c0 + c1 + c17 + c4) + 60*(c13 + c23 + c6 + c7) + 36*(c14 - c16 - c10 + c2 - c26 + c27 - c29 - c34 - c38 + c5) + 12*(c11 - c21 + c32 + c35 - c39 + c41 - c43 - c47 - c49 - c51))
        + u1*u2*u3*( + 288*(-c18 - c3) + 192*(c0 + c1 + c17 + c4) + 144*(-c10 - c25 - c26 - c9) + 96*(c13 + c14 - c16 + c2 + c27 - c29 - c34 - c38 + c5 + c6) + 48*(c23 + c24 + c7 + c8))
        + u1*u_2[3]*( + 132*(-c18 - c3) + 108*(-c10 - c26) + 84*(c0 + c1 + c17 + c4) + 60*(c14 + c24 + c5 + c8) + 36*(c13 - c16 + c2 - c25 + c27 - c29 - c34 - c38 + c6 - c9) + 12*(c12 - c22 + c31 + c36 - c40 + c42 - c44 - c48 - c50 - c52))
        + u_3[2]*( + 32*(c0 + c1 + c13 - c18 - c25 - c3 + c6 - c9) + 16*(c17 + c23 - c29 - c38 - c39 + c4 - c49 + c7) + 8*(c11 + c14 - c16 + c2 - c21 - c26 + c27 + c32 - c34 + c35 + c41 - c43 - c47 + c5 - c51 - c10))
        + u_2[2]*u3*( + 120*(c0 + c1 - c18 - c3) + 72*(c13 + c17 - c25 - c29 - c38 + c4 + c6 - c9) + 48*(c14 - c16 - c10 + c2 - c26 + c27 - c34 + c5) + 24*(c23 - c39 - c49 + c7))
        + u2*u_2[3]*( 120*(c0 + c1 - c18 - c3) + 72*(c14 - c10 + c17 - c26 - c29 - c38 + c4 + c5) + 48*(c13 - c16 + c2 - c25 + c27 - c34 + c6 - c9) + 24*(c24 - c40 - c50 + c8))
        + u_3[3]*( 32*(c0 + c1 - c10 + c14 - c18 - c26 - c3 + c5) + 16*(c17 + c24 - c29 - c38 + c4 - c40 - c50 + c8) + 8*(c12 + c13 - c16 + c2 - c22 - c25 + c27 + c31 - c34 + c36 + c42 - c44 - c48 - c52 + c6 - c9))

        );
    expr.z = 	// g_7(u)
        (
        u_3[0]*( + 48*(c6 - c5) + 24*(c7 - c8 + c9 -c10 + c11 - c12 + c13 - c14) + 8*(c19 - c20 + c21 - c22 + c23 - c24 + c25 - c26 - c31 + c32))
        + u_2[0]*u1*( + 144*(c6 - c5) + 96*(c13 - c14) + 72*(c7 - c8 + c9 -c10) + 48*(c11 - c12) + 36*(c23 - c24 + c25 - c26) + 24*(c32 - c31) + 12*(c19 - c20 + c21 - c22))
        + u_2[0]*u2*( + 144*(c6 - c5) + 120*(c13 + c9) + 72*(c7 - c10 + c11 - c14) + 48*(c25 - c26 - c3 + c32 - c4 - c0 - c1 - c2) + 24*(c21 - c22 + c23 - c24 - c12 - c8))
        + u_2[0]*u3*( + 144*(c6 - c5) + 120*(-c10 - c14) + 72*(c9 - c12 + c13 - c8) + 48*(c0 + c1 + c2 + c25 - c26 + c3 - c31 + c4) + 24*(c11 + c21 - c22 + c23 - c24 + c7) )
        + u0*u_2[1]*( + 138*(c6 - c5) + 114*(c13 - c14) + 66*(c7 - c8 + c9 -c10) + 48*(c23 - c24 + c25 - c26) + 30*(c11 - c12) + 18*(c32 - c31) + 6*(c19 - c20 + c21 - c22 + c35 - c36 + c41 - c42))
        + u0*u1*u2*( + 288*c6 + 264*(c13 - c5) + 216*c9 - 192*c14 + 144*(c25 - c10) + 120*(c7 - c26) + 96*(c11 - c0 - c1) + 72*(c23 - c2 - c3 + c32 - c4) + 48*(- c24 - c8) + 24*(c21 - c22 - c12 - c17 - c18 - c27 + c35 + c41) )
        + u0*u1*u3*( - 288*c5 + 264*(c6 - c14) - 216*c10 + 120*(c25 - c8) + 144*(c9 - c26) + 192*c13 + 96*(c0 + c1 - c12) + 72*(c2 - c24 + c3 - c31 + c4) + 48*(c23 + c7) + 24*(c11 + c17 + c18 + c21 - c22 + c27 - c36 - c42))
        + u0*u_2[2]*( + 144*c6 + 120*(c13 + c9) + 96*(c25 - c0 - c5) + 72*(-c1 - c10 - c14 - c3) + 48*(c11 - c18 - c2 - c26 + c32 - c4 + c7) + 24*(-c16 - c17 + c21 + c23 - c27 - c29 + c35 + c39 + c41 + c43))
        + u0*u2*u3*( + 288*(c6 - c5) + 240*(c13 - c14 - c10 + c9) + 192*(c25 - c26) + 48*(c21 - c22 + c23 - c24 + c11 - c12 + c7 - c8))
        + u0*u_2[3]*( - 144*c5 + 120*(-c10 - c14) + 96*(c0 - c26 + c6) + 72*(c1 + c13 + c3 + c9) + 48*(c18 + c2 - c12 + c25 - c31 + c4 - c8) + 24*(c16 + c17 - c22 - c24 + c27 + c29 - c36 - c40 - c42 - c44))
        + u_3[1]*( + 42*(c13 - c14 - c5 + c6) + 19*(c23 - c24 + c25 - c26 -c10 + c7 - c8 + c9) + 6*(c11 - c12 + c35 - c36) + 4*(c41 - c42 - c31 + c32) + (c19 - c20 + c21 - c22 + c45 - c46 + c47 - c48))
        + u_2[1]*u2*( + 138*(c13 + c6) + 114*(-c14 - c5) + 90*(c25 + c9) + 66*(-c10 - c26) + 48*(c23 - c0 - c1 + c7) + 30*(c11 + c35) + 24*(c32 - c17 - c18 - c2 - c24 - c27 - c3 - c4 + c41 - c8) + 6*(c21 - c22 - c12 - c36 + c47 - c48))
        + u_2[1]*u3*( + 138*(-c14 - c5) + 114*(c13 + c6) + 90*(-c10 - c26) + 66*(c25 + c9) + 48*(c0 + c1 - c24 - c8) + 30*(-c12 - c36) + 24*(c17 + c18 + c2 + c23 + c27 + c3 - c31 + c4 - c42 + c7) + 6*(c11 + c21 - c22 + c35 + c47 - c48))
        + u1*u_2[2]*( + 132*(c13 + c6) + 108*(c25 + c9) + 84*(-c0 - c1 - c14 - c5) + 60*(-c10 - c18 - c26 - c3) + 36*(c11 - c17 - c2 + c23 - c27 + c32 + c35 - c4 + c41 + c7) + 12*(c21 - c16 - c29 - c34 - c38 + c39 + c43 + c47 + c49 + c51))
        + u1*u2*u3*( + 264*(c13 - c14 - c5 + c6) + 216*(c25 - c26 -c10 + c9) + 48*(c23 - c24 + c7 - c8) + 24*(c11 - c12 + c21 - c22 + c35 - c36 + c47 - c48))
        + u1*u_2[3]*( + 132*(-c14 - c5) + 108*(-c10 - c26) + 84*(c0 + c1 + c13 + c6) + 60*(c18 + c25 + c3) + 60*c9 + 36*(c17 - c12 + c2 - c24 + c27 - c31 - c36 + c4 - c42 - c8) + 12*(c16 - c22 + c29 + c34 + c38 - c40 - c44 - c48 - c50 - c52))
        + u_3[2]*( + 32*(c25 - c0 - c1 + c13 - c18 - c3 + c6 + c9) + 16*(c32 - c10 - c14 - c26 + c41 + c43 - c5 + c51) + 8*(c11 - c16 - c17 - c2 + c21 + c23 - c27 - c29 - c34 + c35 - c38 + c39 - c4 + c47 + c49 + c7))
        + u_2[2]*u3*( + 144*(c13 + c25 + c6 + c9) + 96*(-c10 - c14 - c26 - c5) + 48*(-c0 - c1 - c18 - c3) + 24*(c11 - c16 - c17 - c2 + c21 + c23 - c27 - c29 - c34 + c35 - c38 + c39 - c4 + c47 + c49 + c7))
        + u2*u_2[3]*( + 144*(-c10 - c14 - c26 - c5) + 96*(c13 + c25 + c6 + c9) + 48*(c0 + c1 + c18 + c3) + 24*(c16 + c17 - c12 + c2 - c22 - c24 + c27 + c29 + c34 - c36 + c38 + c4 - c40 - c48 - c50 - c8))
        + u_3[3]*(
            32*(c0 + c1 - c10 - c14 + c18 - c26 + c3 - c5)
            + 16*(c13 + c25 - c31 - c42 - c44 - c52 + c6 + c9)
            + 8*(c16 + c17 - c12 + c2 - c22 - c24 + c27 + c29 + c34 - c36 + c38 + c4 - c40 - c48 - c50 - c8)
            )
        );
    return expr*SCALE_Tj;
}

vec3 compute_gradient(vec3 p_in)
{
    vec3 gj = eval_gj();
    // Instead of using a lookup table corresponding to the first three columns of Table 2,
    // we can convert the formulae as follows.: (f_5,f_6,f_7) = (R^T)*(P^T)*(g_5,g_6,g_7)
    // Note that, since mat3 is in column-major format, the following formula results in P^T*gj not P*gj.
    return vec3(dot(P[0],gj), dot(P[1],gj), dot(P[2],gj))*(2*type_R-1)*scale_axes;  
}

#define E12 2
#define E13 0
#define E14 4
#define E23 5
#define E24 3
#define E34 1

// Evaluate six 2nd derivtive formulae.
float[6] eval_fij(void)
{
    float   expr[6];
    vec4    u_2 = u*u;
    expr[E12] =     // g_12(u)
    (
        u_2[0]*(  8*(c5 + c6 - c16 - c17) + 4*c0 + 2*(c19 + c20 - c21 - c22 - c23 - c24 + c25 + c26 - c27 - c28 - c29 - c30 + c31 + c32))
        + u0*u1*(12*(c5 + c6 - c16 - c17) + 8*c0 + 4*(c13 + c14 - c2 - c21 - c22 - c23 - c24 + c25 + c26 - c27 - c29 - c3 + c31 + c32 - c33 - c37 + c7 + c8))
        + u0*u2*(16*(c5 + c6 - c16 - c17) + 8*(c0 - c21 - c23 + c25 + c26 - c27 - c29 + c32))
        + u0*u3*(16*(c5 + c6 - c16 - c17) + 8*(c0 - c22 - c24 + c25 + c26 - c27 - c29 + c31))
        + u_2[1]*(4*(c13 + c14 - c16 - c17 - c3 - c33 + c5 + c6) + 2*(c0 + c1 - c2 + c25 + c26 - c27 - c29 - c37 + c7 + c8) + (c31 + c32 - c10 - c21 - c22 - c23 - c24 + c41 + c42 - c45 - c46 - c9))
        + u1*u2*(12*(c5 + c6 - c16 - c17) + 8*(c25 + c26 - c27 - c29) + 4*(c0 + c1 + c13 + c14 - c21 - c23 - c3 + c32 - c33 + c41 - c45 - c9))
        + u1*u3*(12*(c5 + c6 - c16 - c17) + 8*(c25 + c26 - c27 - c29) + 4*(c0 + c1 - c10 + c13 + c14 - c22 - c24 - c3 + c31 - c33 + c42 - c46))
        + u_2[2]*(4*(c0 - c16 - c17 + c18 - c21 - c23 + c25 + c26 - c27 - c29 + c32 - c35 - c39 + c5 + c51 + c6))
        + u2*u3*(16*(-c16 - c17 + c25 + c26 - c27 - c29 + c5 + c6))
        + u_2[3]*(4*(c0 - c16 - c17 + c18 - c22 - c24 + c25 + c26 - c27 - c29 + c31 - c36 - c40 + c5 + c52 + c6))
    )*SCALE_Tij;

    expr[E13] =     // g_13(u)
    (
        u_2[0]*(4*c0 - 8*c12 - 8*c13 + 2*c19 - 2*c20 + 2*c21 - 2*c22 - 2*c23 + 2*c24 - 2*c25 + 2*c26 - 2*c27 - 2*c28 + 2*c29 + 8*c3 + 2*c30 - 2*c31 - 2*c32 + 8*c4)
        + u0*u1*(8*c0 - 12*c12 - 12*c13 + 4*c17 + 4*c18 - 4*c2 - 4*c20 - 4*c22 - 4*c23 + 4*c24 - 4*c25 + 4*c26 - 4*c27 + 4*c29 + 12*c3 + 4*c30 - 4*c31 - 4*c35 + 12*c4 - 4*c41 - 4*c5 + 4*c7 + 4*c9)
        + u0*u2*(8*c0 - 8*c12 - 8*c13 + 8*c17 + 8*c18 - 8*c2 - 8*c22 - 8*c25 + 8*c29 + 8*c3 - 8*c35 + 8*c4 - 8*c41 - 8*c5 + 8*c7 + 8*c9)
        + u0*u3*(8*c0 - 16*c12 - 16*c13 - 8*c22 + 8*c24 - 8*c25 + 8*c26 - 8*c27 + 8*c29 + 16*c3 - 8*c31 + 16*c4)
        + u_2[1]*(2*c0 + 2*c1 - c10 - 4*c12 - 4*c13 + 4*c17 + 4*c18 - 2*c2 - c20 - c22 - c23 + 2*c24 - c25 + 2*c26 - 2*c27 + c29 + 4*c3 + c30 - 2*c31 - 4*c35 + c37 + c38 + 4*c4 - 2*c41 - c45 - c47 - 4*c5 + 2*c7 - c8 + 2*c9)
        + u1*u2*(4*c0 + 4*c1 - 4*c10 - 4*c12 - 4*c13 + 12*c17 + 12*c18 - 8*c2 - 4*c22 - 4*c25 + 4*c29 + 4*c3 - 12*c35 + 4*c38 + 4*c4 - 8*c41 - 4*c47 - 12*c5 + 8*c7 + 8*c9)
        + u1*u3*(4*c0 + 4*c1 - 4*c10 - 12*c12 - 12*c13 + 4*c17 + 4*c18 - 4*c22 + 8*c24 - 4*c25 + 8*c26 - 8*c27 + 4*c29 + 12*c3 - 8*c31 - 4*c35 + 4*c38 + 12*c4 - 4*c47 - 4*c5)
        + u_2[2]*(4*c1 - 4*c10 - 4*c16 + 4*c17 + 4*c18 - 4*c2 - 4*c35 + 4*c38 + 4*c39 - 4*c41 - 4*c47 - 4*c5 - 4*c51 + 4*c6 + 4*c7 + 4*c9)
        + u2*u3*(8*c0 + 8*c1 - 8*c10 - 8*c12 - 8*c13 + 8*c17 + 8*c18 - 8*c22 - 8*c25 + 8*c29 + 8*c3 - 8*c35 + 8*c38 + 8*c4 - 8*c47 - 8*c5)
        + u_2[3]*(4*c0 - 4*c12 - 4*c13 + 4*c14 - 4*c22 + 4*c24 - 4*c25 + 4*c26 - 4*c27 + 4*c29 + 4*c3 - 4*c31 - 4*c34 + 4*c4 - 4*c44 + 4*c50)
    )*SCALE_Tij;
    expr[E14] =     // g_14(u)
    (
        u_2[0]*( + 8*(c1 - c10 + c2 - c7) + 4*c0 + 2*(c20 + c21 - c22 - c23 + c24 + c25 - c26 + c27 + c28 - c29 - c30 - c31 - c32 - c19))
        + u0*u1*( + 16*(c1 - c10 + c2 - c7) + 8*(c0 - c23 - c26 + c27) + 4*(c20 + c21 + c24 + c25 - c29 - c30 - c31 - c32) )
        + u0*u2*( + 16*(c1 - c10 + c2 - c7) + 8*(c0 + c21 - c23 + c25 - c26 + c27 - c29 - c32))
        + u0*u3*(8*(c0 + c1 - c10 + c12 + c14 + c16 + c18 + c2 - c23 - c26 + c27 - c4 - c40 - c44 - c6 - c7))
        + u_2[1]*( + 6*(c0 + c1 - c10 + c2 - c23 - c26 + c27 - c7) + (c20 + c21 + c24 + c25 - c29 - c30 - c31 - c32 - c37 - c38 - c41 - c42 + c46 + c47 + c8 + c9))
        + u1*u2*( 12*(c0 + c1 - c10 + c2 - c23 - c26 + c27 - c7) + 4*(c21 + c25 - c29 - c32 - c38 - c41 + c47 + c9))
        + u1*u3*( 8*(c0 + c1 - c10 + c2 - c23 - c26 + c27 - c7) + 4*(c12 - c13 + c14 + c16 - c17 + c18 + c3 + c34 + c36 - c4 - c40 - c44 + c5 - c50 - c52 - c6)) 
        + u_2[2]*(4*(c0 + c1 - c10 + c2 + c21 - c23 + c25 - c26 + c27 - c29 - c32 - c38 - c41 + c47 - c7 + c9))
        + u2*u3*(8*(c0 + c1 - c10 - c13 + c16 + c18 + c2 - c23 - c26 + c27 + c3 + c34 - c40 - c50 - c6 - c7))
        + u_2[3]*(4*(c12 - c13 + c14 + c16 - c17 + c18 + c3 + c34 + c36 - c4 - c40 - c44 + c5 - c50 - c52 - c6))
    )*SCALE_Tij;
    expr[E23] =     // g_23(u)
    (
        u_2[0]*( + 8*(c1 + c2 - c8 - c9) + 4*c0 + 2*(c19 - c20 - c21 + c22 + c23 - c24 - c25 + c26 + c27 + c28 - c29 - c30 - c31 - c32)) 
        + u0*u1*( + 16*(c1 + c2 - c8 - c9) + 8*(c0 - c24 - c25 + c27) + 4*(c19 + c22 + c23 + c26 - c29 - c30 - c31 - c32)) 
        + u0*u2*(8*(c0 + c1 + c11 + c13 + c16 + c18 + c2 - c24 - c25 + c27 - c39 - c4 - c43 - c5 - c8 - c9))
        + u0*u3*( + 16*(c1 + c2 - c8 - c9) + 8*(c0 + c22 - c24 - c25 + c26 + c27 - c29 - c31)) 
        + u_2[1]*( + 6*(c0 + c1 + c2 - c24 - c25 + c27 - c8 - c9) + (c10 + c19 + c22 + c23 + c26 - c29 - c30 - c31 - c32 - c37 - c38 - c41 - c42 + c45 + c48 + c7))
        + u1*u2*( + 8*(c0 + c1 + c2 - c24 - c25 + c27 - c8 - c9) + 4*(c11 + c13 - c14 + c16 - c17 + c18 + c3 + c34 + c35 - c39 - c4 - c43 - c49 - c5 - c51 + c6))
        + u1*u3*( 12*(c0 + c1 + c2 - c24 - c25 + c27 - c8 - c9) + 4*(c10 + c22 + c26 - c29 - c31 - c38 - c42 + c48)) 
        + u_2[2]*(4*(c11 + c13 - c14 + c16 - c17 + c18 + c3 + c34 + c35 - c39 - c4 - c43 - c49 - c5 - c51 + c6))
        + u2*u3*(8*(c0 + c1 - c14 + c16 + c18 + c2 - c24 - c25 + c27 + c3 + c34 - c39 - c49 - c5 - c8 - c9))
        + u_2[3]*(4*(c0 + c1 + c10 + c2 + c22 - c24 - c25 + c26 + c27 - c29 - c31 - c38 - c42 + c48 - c8 - c9))
    )*SCALE_Tij;
    expr[E24] =     // g_24(u)
    (
        u_2[0]*(  8*(c3 + c4 - c11 - c14) + 4*c0 + 2*(c30 - c31 - c32 - c19 + c20 - c21 + c22 + c23 - c24 + c25 - c26 - c27 - c28 + c29))
        + u0*u1*(12*(c3 + c4 - c11 - c14) + 8*c0 + 4*(c10 + c17 + c18 - c19 - c2 - c21 + c23 - c24 + c25 - c26 - c27 + c29 + c30 - c32 - c36 - c42 - c6 + c8))
        + u0*u2*(16*(c3 + c4 - c11 - c14) + 8*(c0 - c21 + c23 + c25 - c26 - c27 + c29 - c32))
        + u0*u3*( 8*(c0 + c10 - c11 - c14 + c17 + c18 - c2 - c21 - c26 + c29 + c3 - c36 + c4 - c42 - c6 + c8))
        + u_2[1]*( + 4*(c3 - c11 - c14 + c17 + c18 - c36 + c4 - c6) + 2*(c0 + c1 + c10 - c2 + c23 + c25 - c27 - c32 - c42 + c8) + (c37 + c38 - c19 - c21 - c24 - c26 + c29 + c30 - c46 - c48 - c7 - c9))
        + u1*u2*( + 12*(c3 - c11 - c14 + c4) + 8*(c23 + c25 - c27 - c32) + 4*(c0 + c1 + c17 + c18 - c21 - c26 + c29 - c36 + c38 - c48 - c6 - c9))
        + u1*u3*( + 12*(c17 + c18 - c36 - c6) + 8*(c10 - c2 - c42 + c8) + 4*(c0 + c1 - c11 - c14 - c21 - c26 + c29 + c3 + c38 + c4 - c48 - c9))
        + u_2[2]*(4*(c0 - c11 + c13 - c14 - c21 + c23 + c25 - c26 - c27 + c29 + c3 - c32 - c34 + c4 - c43 + c49))
        + u2*u3*(8*(c0 + c1 - c11 - c14 + c17 + c18 - c21 - c26 + c29 + c3 - c36 + c38 + c4 - c48 - c6 - c9))
        + u_2[3]*(4*(c1 + c10 - c16 + c17 + c18 - c2 - c36 + c38 + c40 - c42 - c48 + c5 - c52 - c6 + c8 - c9))
    )*SCALE_Tij;
    expr[E34] =     // g_34(u)
    (
        u_2[0]*( 8*(c5 + c6 - c15 - c18) + 4*c0 + 2*(c21 + c22 + c23 + c24 - c25 - c26 - c27 - c28 - c29 - c30 + c31 + c32 - c19 - c20))
        + u0*u1*( 12*(c5 + c6 - c15 - c18) + 8*c0 + 4*(c10 + c13 + c14 - c19 - c2 - c20 + c23 + c24 - c25 - c26 - c27 - c30 + c31 + c32 - c34 - c38 - c4 + c9))
        + u0*u2*( 8*(c0 + c10 + c13 + c14 - c15 - c18 - c19 - c2 - c25 + c32 - c34 - c38 - c4 + c5 + c6 + c9) )
        + u0*u3*(8*(c0 + c10 + c13 + c14 - c15 - c18 - c2 - c20 - c26 + c31 - c34 - c38 - c4 + c5 + c6 + c9))
        + u_2[1]*( 4*(c13 + c14 - c15 - c18 - c34 - c4 + c5 + c6) + 2*(c0 + c1 + c10 - c2 + c23 + c24 - c27 - c30 - c38 + c9) + (c31 + c32 - c19 - c20 - c25 - c26 + c41 + c42 - c47 - c48 - c7 - c8))
        + u1*u2*( 12*(c13 + c14 - c34 - c4) + 8*(c10 - c2 - c38 + c9) + 4*(c0 + c1 - c15 - c18 - c19 - c25 + c32 + c41 - c47 + c5 + c6 - c7))
        + u1*u3*( 12*(c13 + c14 - c34 - c4) + 8*(c10 - c2 - c38 + c9) + 4*(c0 + c1 - c15 - c18 - c20 - c26 + c31 + c42 - c48 + c5 + c6 - c8))
        + u_2[2]*( 4*(c1 + c10 - c11 + c13 + c14 - c2 + c3 - c34 - c38 - c4 + c41 + c43 - c47 - c49 - c7 + c9))
        + u2*u3*(16*(c10 + c13 + c14 - c2 - c34 - c38 - c4 + c9))
        + u_2[3]*(4*(c1 + c10 - c12 + c13 + c14 - c2 + c3 - c34 - c38 - c4 + c42 + c44 - c48 - c50 - c8 + c9))
    )*SCALE_Tij;
    return expr;
}

// Mapping of the formulae f_ij (last six columns of Table 2)
const int map_gij[24*6] = 
int[]
(
    E12, E13, E14, E23, E24, E34,   //  0
    E13, E12, E14, E23, E34, E24,
    E12, E23, E24, E13, E14, E34,
    E23, E12, E24, E13, E34, E14,
    E23, E13, E34, E12, E24, E14,   
    E13, E23, E34, E12, E14, E24,

    E34, E24, E14, E23, E13, E12,   //  6
    E24, E34, E14, E23, E12, E13,
    E34, E14, E24, E13, E23, E12,
    E14, E34, E24, E13, E12, E23,
    E14, E24, E34, E12, E13, E23,
    E24, E14, E34, E12, E23, E13,   

    E34, E13, E23, E14, E24, E12,   // 12
    E24, E12, E23, E14, E34, E13,
    E34, E23, E13, E24, E14, E12,
    E14, E12, E13, E24, E34, E23,   
    E14, E13, E12, E34, E24, E23,
    E24, E23, E12, E34, E14, E13,

    E12, E24, E23, E14, E13, E34,   // 18
    E13, E34, E23, E14, E12, E24,
    E12, E14, E13, E24, E23, E34,   
    E23, E34, E13, E24, E12, E14,
    E23, E24, E12, E34, E13, E14,
    E13, E14, E12, E34, E23, E24

);

float[6] compute_Hessian(void)
{
    float    expr[6] = eval_fij();

    float d12 = expr[map_gij[6*idx + 0]];   // D_{xi1,xi2}f
    float d13 = expr[map_gij[6*idx + 1]];   // D_{xi1,xi3}f
    float d14 = expr[map_gij[6*idx + 2]];   // D_{xi1,xi4}f
    float d23 = expr[map_gij[6*idx + 3]];   // D_{xi2,xi3}f
    float d24 = expr[map_gij[6*idx + 4]];   // D_{xi2,xi4}f
    float d34 = expr[map_gij[6*idx + 5]];   // D_{xi3,xi4}f

    // Convert to Dxx,Dyy,Dzz,Dxy,Dxz,Dyz
    float Dxx = 0.25*(-d12-d13        -d24-d34)*(scale_axes.x*scale_axes.x);
    float Dyy = 0.25*(-d12    -d14-d23    -d34)*(scale_axes.y*scale_axes.y);
    float Dzz = 0.25*(    -d13-d14-d23-d24    )*(scale_axes.z*scale_axes.z);
    float Dxy = 0.25*( d12                -d34)*(scale_axes.x*scale_axes.y);
    float Dxz = 0.25*(     d13        -d24    )*(scale_axes.x*scale_axes.z);
    float Dyz = 0.25*(        -d14+d23        )*(scale_axes.y*scale_axes.z);
    return float[6](Dxx,Dyy,Dzz,Dxy,Dxz,Dyz);
}

void main() 
{
    vec3 start = texture(tex_front, vTexCoord).xyz*dim; // starting position of the ray
    vec3 end = texture(tex_back, vTexCoord).xyz*dim;    // ending position of the ray
    
    vec3 p = start; // current position of the ray
    vec3 p_prev;    // previous position of the ray
    vec3 dir = normalize(end-start);    // (normalized) ray direction
    
    float step = scale_step*dim.x;  // ray step size
    
    float len = 0;  // accumulated length of the ray
    float len_full = length(end - start);   // full length of the ray
    float voxel; // (reconstructed) value of the volume at the current position
    float voxel_prev;    // (reconstructed) value of the volume at the previous position

    ///////////////////////////////////////////////////////////////////////////////////

    // To determine if we're hitting the front or back face of the isosurface, 
    // we first evaluate the value at the starting position.
    // voxel<level --> we're hitting the front face
    // voxel>=level --> we're hitting the back face
    voxel = EVAL(p);
    float   orientation = 2.0*float(voxel < level)-1.0;     // equivalent to (voxel<level?1:-1)

    for(int i = 0 ; i < 1000 ; i++)
    {
        p += step*dir;  // Step forward
        len += step;    // Accumulate the length

        if(len > len_full) discard; // We're out of the bounding box. Let's finish.
        
        voxel = EVAL(p);    // Evaluate the value at the current position.
        
        if(orientation*voxel > orientation*level)   // If we just crossed the isosurface...
        {
            if(abs(voxel-voxel_prev) > 0.00001) // For stable computation...
            {
                // Perform one step of Regula Falsi.
                p = (p*(voxel_prev-level) - p_prev*(voxel-level))/(voxel_prev-voxel);
                preprocess(p);
                fetch_coefficients();
            }
            
            vec3    g = compute_gradient(p);    // Compute the gradient
            float[6] h = compute_Hessian(); // Compute the Hessian

            // Store the position/gradient/Hessian using MRT (multi-render target) feature.
            fPosition = vec4(p/scale_axes - vec3(.5), orientation);
            fGradient = vec4(g,0);
            fHessian1 = vec4(h[0], h[1], h[2], 0);
            fHessian2 = vec4(h[3], h[4], h[5], 0);
            return;
        }
        // Update the `previous' info.
        voxel_prev = voxel;
        p_prev = p;
    }
    discard;
}


