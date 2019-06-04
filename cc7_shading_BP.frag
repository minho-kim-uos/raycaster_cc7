#version 330 core

in vec2 vTexCoord;
out vec4 fColor;
uniform sampler2D tex_position;
uniform sampler2D tex_gradient;
uniform mat4        MV;

struct TMaterial
{
	vec3	ambient;
	vec3	diffuse;
	vec3	specular;
	vec3	emission;
	float	shininess;
};
struct TLight
{
	vec4	position;
	vec3	ambient;
	vec3	diffuse;
	vec3	specular;
};

TMaterial uMaterial[2] =
TMaterial[2]
(
    // front face: silver
    TMaterial(
        vec3(0.19225,0.19225,0.19225), 
        vec3(0.50754,0.50754,0.50754),
        vec3(0.508273,0.508273,0.508273), 
        vec3(0,0,0), 
        0.4*128.0), 
    // back face: red plastic
    TMaterial(
        vec3(0, 0, 0),
        vec3(0.5, 0.0, 0.0),
        vec3(0.7, 0.6, 0.6),
        vec3(0,0,0),
        .25*128.0)
);

TLight uLight = TLight(
        vec4(1,1,1,0),
        vec3(.2,.2,.2),
        vec3(1,1,1),
        vec3(1,1,1)
        );

vec4 shade_Blinn_Phong(vec3 n, vec4 pos_eye, TMaterial material, TLight light)
{
    vec3    l;
    if(light.position.w == 1.0)
        l = normalize((light.position - pos_eye).xyz);  // positional light
    else
        l = normalize((light.position).xyz);    // directional light
    vec3    v = -normalize(pos_eye.xyz);
    vec3    h = normalize(l + v);
    float   l_dot_n = max(dot(l, n), 0.0);
    vec3    ambient = light.ambient * material.ambient;
    vec3    diffuse = light.diffuse * material.diffuse * l_dot_n;
    vec3    specular = vec3(0.0);
    if(l_dot_n >= 0.0)
    {
        specular = light.specular * material.specular * pow(max(dot(h, n), 0.0), material.shininess);
    }
    return vec4(ambient + diffuse + specular, 1);
}

void main() {
    vec4 p = texture(tex_position, vTexCoord).xyzw;
    vec3 g = -normalize(texture(tex_gradient, vTexCoord).xyz);
    if(p.w!=0.0)
        fColor = shade_Blinn_Phong(normalize(mat3(MV)*(p.w*g)), MV*vec4(p.xyz,1), uMaterial[int(0.5*(1.0-p.w))], uLight);
    else
        discard;
}

