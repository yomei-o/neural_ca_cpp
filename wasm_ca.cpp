// WebAssembly runtime for the trained Neural CA (inference only, no autograd).
// Loads the learned per-cell rule (nca.bin) and steps the grid so a seed grows
// into the target and self-heals when damaged. JS calls ca_step() each frame and
// reads ca_rgba() to paint the canvas; clicks call ca_damage().
#include <emscripten.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

static int N = 32, C = 16, H = 128;
static std::vector<float> W1, b1, W2, b2;   // MLP: (H,3C) (H) (C,H) (C)
static std::vector<float> S;                // state C*N*N
static std::vector<uint8_t> RGBA;           // N*N*4 for the canvas
static uint64_t rng = 0x2545F4914F6CDD1Dull;
static inline float frand(){ rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return (rng>>11)*(1.0f/9007199254740992.0f); }

static inline int SN(){ return N*N; }
static inline float at(int c,int x,int y){ if(x<0||y<0||x>=N||y>=N) return 0.f; return S[(size_t)c*SN()+y*N+x]; }

extern "C" {

EMSCRIPTEN_KEEPALIVE
int ca_init(){
    FILE* f=std::fopen("nca.bin","rb"); if(!f) return 0;
    int meta[3]; if(std::fread(meta,4,3,f)!=3){ std::fclose(f); return 0; }
    N=meta[0]; C=meta[1]; H=meta[2];
    auto rd=[&](std::vector<float>& v){ int n=0; std::fread(&n,4,1,f); v.resize(n); std::fread(v.data(),4,n,f); };
    rd(W1); rd(b1); rd(W2); rd(b2);
    std::fclose(f);
    S.assign((size_t)C*N*N, 0.f);
    RGBA.assign((size_t)N*N*4, 0);
    return N;
}

EMSCRIPTEN_KEEPALIVE int ca_n(){ return N; }

EMSCRIPTEN_KEEPALIVE
void ca_reset(){
    std::fill(S.begin(), S.end(), 0.f);
    int cx=N/2, cy=N/2, i=cy*N+cx;
    for(int c=3;c<C;++c) S[(size_t)c*SN()+i]=1.f;    // seed: alpha+hidden = 1
}

EMSCRIPTEN_KEEPALIVE
void ca_damage(int px,int py,int r){
    for(int y=0;y<N;++y) for(int x=0;x<N;++x){ int dx=x-px,dy=y-py;
        if(dx*dx+dy*dy<=r*r) for(int c=0;c<C;++c) S[(size_t)c*SN()+y*N+x]=0.f; }
}

EMSCRIPTEN_KEEPALIVE
void ca_step(){
    const int P=3*C, sn=SN();
    std::vector<float> dx((size_t)C*sn, 0.f);
    std::vector<float> perc(P), h(H);
    for(int y=0;y<N;++y) for(int x=0;x<N;++x){
        // perception: identity, sobel-x, sobel-y per channel
        for(int c=0;c<C;++c){
            float id=at(c,x,y);
            float gx=(at(c,x+1,y-1)+2*at(c,x+1,y)+at(c,x+1,y+1) - at(c,x-1,y-1)-2*at(c,x-1,y)-at(c,x-1,y+1))/8.f;
            float gy=(at(c,x-1,y+1)+2*at(c,x,y+1)+at(c,x+1,y+1) - at(c,x-1,y-1)-2*at(c,x,y-1)-at(c,x+1,y-1))/8.f;
            perc[c]=id; perc[C+c]=gx; perc[2*C+c]=gy;
        }
        for(int j=0;j<H;++j){ float s=b1[j]; const float* w=&W1[(size_t)j*P];
            for(int k=0;k<P;++k) s+=w[k]*perc[k]; h[j]=s>0?s:0; }
        for(int c=0;c<C;++c){ float s=b2[c]; const float* w=&W2[(size_t)c*H];
            for(int j=0;j<H;++j) s+=w[j]*h[j]; dx[(size_t)c*sn+y*N+x]=s; }
    }
    // stochastic update (per cell), then alive mask
    for(int i=0;i<sn;++i){ if(frand()<0.5f) continue; for(int c=0;c<C;++c) S[(size_t)c*sn+i]+=dx[(size_t)c*sn+i]; }
    std::vector<uint8_t> alive(sn,0);
    for(int y=0;y<N;++y) for(int x=0;x<N;++x){ float mx=0.f;
        for(int dyy=-1;dyy<=1;++dyy)for(int dxx=-1;dxx<=1;++dxx){int yy=y+dyy,xx=x+dxx; if(yy<0||yy>=N||xx<0||xx>=N)continue; mx=std::max(mx,S[(size_t)3*sn+yy*N+xx]);}
        alive[y*N+x]= mx>0.1f?1:0; }
    for(int i=0;i<sn;++i) if(!alive[i]) for(int c=0;c<C;++c) S[(size_t)c*sn+i]=0.f;
}

// composite premultiplied RGBA over white -> opaque RGBA bytes for the canvas
EMSCRIPTEN_KEEPALIVE
uint8_t* ca_rgba(){
    int sn=SN();
    for(int i=0;i<sn;++i){ float a=std::min(1.f,std::max(0.f,S[(size_t)3*sn+i]));
        for(int c=0;c<3;++c){ float v=S[(size_t)c*sn+i]+(1.f-a); v=std::min(1.f,std::max(0.f,v));
            RGBA[i*4+c]=(uint8_t)(v*255.f+0.5f); }
        RGBA[i*4+3]=255; }
    return RGBA.data();
}

}  // extern "C"
