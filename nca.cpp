// Neural Cellular Automata (from-scratch autograd, CPU) with POOL + DAMAGE
// training, so the grown pattern is persistent (holds its shape when the rule
// keeps running) and self-healing (regrows after part of it is erased).
//
// Each cell has C channels (0..3 = RGBA premultiplied, rest hidden). One learned
// per-cell rule (perceive neighbourhood -> tiny MLP -> state delta) is applied to
// every cell in parallel, stochastically, with an "alive" mask. Trained by
// running B states T steps and matching the target image; a persistent pool of
// states plus random damage teaches stability and regeneration.
#include "autograd.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

using namespace ag;

static int N = 28;            // grid side
static const int C = 16;      // channels
static int H = 128;           // MLP hidden
static int B = 4;             // batch
static int POOL = 256;
static int TMIN = 48, TMAX = 64;

// ---- fixed perception: identity, sobel-x, sobel-y per channel -> 3C outputs ---
static Tensor make_perception() {
    Tensor w = Tensor::zeros({3 * C, C, 3, 3}, false);
    float idn[9]={0,0,0,0,1,0,0,0,0}, sx[9]={-1,0,1,-2,0,2,-1,0,1}, sy[9]={-1,-2,-1,0,0,0,1,2,1};
    for (int i=0;i<9;++i){ sx[i]/=8.f; sy[i]/=8.f; }
    auto& d = w.data();
    for (int g=0; g<3; ++g) for (int c=0; c<C; ++c) {
        int o=g*C+c; float* f=(g==0)?idn:(g==1)?sx:sy;
        for (int k=0;k<9;++k) d[((o*C+c)*3 + k/3)*3 + k%3] = f[k];
    }
    return w;
}

// ---- default target: a friendly generic smiley (no copyrighted artwork).
//      Premultiplied RGBA, size 4*N*N. Overridable via a raw file (--target). ---
static std::vector<float> make_target_face() {
    std::vector<float> t((size_t)4*N*N, 0.f);
    float cx=N/2.f-0.5f, cy=N/2.f-0.5f, R=N*0.42f;
    auto set=[&](int x,int y,float r,float g,float b){ if(x<0||y<0||x>=N||y>=N)return;
        int i=y*N+x; t[0*N*N+i]=r; t[1*N*N+i]=g; t[2*N*N+i]=b; t[3*N*N+i]=1.f; };
    for (int y=0;y<N;++y) for (int x=0;x<N;++x){ float dx=x-cx,dy=y-cy;
        if (dx*dx+dy*dy<=R*R) set(x,y,1.0f,0.82f,0.0f); }          // yellow face
    float ex=N*0.16f, ey=-N*0.10f, er=N*0.06f;
    for (int s=-1;s<=1;s+=2) for (int y=0;y<N;++y) for (int x=0;x<N;++x){
        float dx=x-(cx+s*ex), dy=y-(cy+ey); if (dx*dx+dy*dy<=er*er) set(x,y,0.05f,0.03f,0.03f); } // eyes
    for (int x=0;x<N;++x) for (int y=0;y<N;++y){                     // smile arc
        float dx=(x-cx)/(N*0.24f), dy=(y-(cy+N*0.06f))/(N*0.20f), rr=dx*dx+dy*dy;
        if (rr>0.55f && rr<0.95f && (y-cy)>N*0.05f) set(x,y,0.05f,0.03f,0.03f); }
    return t;
}

static std::vector<float> load_target(const std::string& path) {
    // raw format: first two ints = w,h (must equal N), then w*h*4 floats RGBA premult.
    FILE* f=std::fopen(path.c_str(),"rb"); if(!f) return {};
    int w=0,h=0; if(std::fread(&w,4,1,f)!=1||std::fread(&h,4,1,f)!=1||w!=N||h!=N){ std::fclose(f); return {}; }
    std::vector<float> t((size_t)4*N*N); std::fread(t.data(),4,t.size(),f); std::fclose(f); return t;
}

static void write_ppm(const char* path, const std::vector<float>& s, int base) {
    FILE* f=std::fopen(path,"wb"); if(!f) return;
    std::fprintf(f,"P6\n%d %d\n255\n",N,N);
    for(int i=0;i<N*N;++i){ float a=std::min(1.f,std::max(0.f,s[base+3*N*N+i]));
        for(int c=0;c<3;++c){ float v=s[base+c*N*N+i]+(1.f-a); v=std::min(1.f,std::max(0.f,v));
            std::fputc((unsigned char)(v*255.f+0.5f),f); } }
    std::fclose(f);
}

static void ascii_alpha(const std::vector<float>& s, int base) {
    const char* ramp=" .:-=+*#%@";
    for (int y=0;y<N;++y){ std::string line;
        for (int x=0;x<N;++x){ float a=s[base+3*N*N+y*N+x]; int l=(int)(std::max(0.f,std::min(1.f,a))*9); line+=ramp[l]; }
        std::printf("%s\n", line.c_str()); }
}

int main(int argc, char** argv) {
    int iters=2000; float lr=2e-3f; std::string tpath, out="nca.bin";
    for (int i=1;i+1<argc;i+=2){ std::string k=argv[i],v=argv[i+1];
        if(k=="--n")N=std::atoi(v.c_str()); else if(k=="--iters")iters=std::atoi(v.c_str());
        else if(k=="--lr")lr=(float)std::atof(v.c_str()); else if(k=="--h")H=std::atoi(v.c_str());
        else if(k=="--b")B=std::atoi(v.c_str()); else if(k=="--target")tpath=v;
        else if(k=="--out")out=v; else if(k=="--tmax")TMAX=std::atoi(v.c_str()); }
    TMIN=std::max(24,TMAX-16);
    seed(1);

    Tensor Wp=make_perception(), bp=Tensor::zeros({3*C},false);
    Tensor W1=Tensor::randn({H,3*C,1,1}, std::sqrt(2.f/(3*C)), true), b1=Tensor::zeros({H},true);
    Tensor W2=Tensor::zeros({C,H,1,1},true), b2=Tensor::zeros({C},true);
    std::vector<Tensor> params={W1,b1,W2,b2};
    std::vector<std::vector<float>> mA(params.size()), vA(params.size());
    for(size_t i=0;i<params.size();++i){ mA[i].assign(params[i].numel(),0.f); vA[i].assign(params[i].numel(),0.f); }
    float b1a=0.9f,b2a=0.999f,eps=1e-8f; int at=0;

    std::vector<float> target = tpath.empty()? make_target_face() : load_target(tpath);
    if (target.empty()){ std::printf("target load failed (%s); using face\n", tpath.c_str()); target=make_target_face(); }

    const int cn=C*N*N, sn=N*N;
    // batch target + channel mask
    Tensor tgt=Tensor::zeros({B,C,N,N},false), chm=Tensor::zeros({B,C,N,N},false);
    for(int b=0;b<B;++b) for(int c=0;c<4;++c) for(int i=0;i<sn;++i){
        tgt.data()[(b*C+c)*sn+i]=target[c*sn+i]; chm.data()[(b*C+c)*sn+i]=1.f; }

    auto seed_into=[&](std::vector<float>& st){ std::fill(st.begin(),st.end(),0.f);
        int i=(N/2)*N+(N/2); for(int c=3;c<C;++c) st[c*sn+i]=1.f; };
    auto rgba_mse=[&](const std::vector<float>& st)->float{ float e=0; for(int c=0;c<4;++c) for(int i=0;i<sn;++i){ float d=st[c*sn+i]-target[c*sn+i]; e+=d*d; } return e/(4*sn); };
    auto damage=[&](std::vector<float>& st){ float rx=randf()*N, ry=randf()*N, rr=N*(0.12f+randf()*0.16f);
        for(int y=0;y<N;++y)for(int x=0;x<N;++x){ float dx=x-rx,dy=y-ry; if(dx*dx+dy*dy<rr*rr){ for(int c=0;c<C;++c) st[c*sn+y*N+x]=0.f; } } };

    std::vector<std::vector<float>> pool(POOL, std::vector<float>(cn,0.f));
    for(auto& s:pool) seed_into(s);

    auto save_weights=[&](const std::string& path){ FILE* f=std::fopen(path.c_str(),"wb"); if(!f) return;
        int meta[3]={N,C,H}; std::fwrite(meta,4,3,f);
        for(auto& t:std::vector<Tensor>{W1,b1,W2,b2}){ int n=t.numel(); std::fwrite(&n,4,1,f); std::fwrite(t.data().data(),4,n,f);} std::fclose(f); };
    auto preview=[&](const std::string& path){ std::vector<float> st(cn,0.f); seed_into(st);
        Tensor s2=Tensor::zeros({1,C,N,N},false);
        for(int step=0;step<TMAX;++step){ std::copy(st.begin(),st.end(),s2.data().begin());
            Tensor pc=conv2d(s2,Wp,bp,1,1); Tensor h=relu(conv2d(pc,W1,b1,1,0)); Tensor dx=conv2d(h,W2,b2,1,0);
            auto& sd=s2.data(); for(int i=0;i<sn;++i){ if(randf()<0.5f) continue; for(int c=0;c<C;++c) sd[c*sn+i]+=dx.data()[c*sn+i]; }
            for(int y=0;y<N;++y)for(int x=0;x<N;++x){ float mx=0.f; for(int dy=-1;dy<=1;++dy)for(int dx2=-1;dx2<=1;++dx2){int yy=y+dy,xx=x+dx2; if(yy<0||yy>=N||xx<0||xx>=N)continue; mx=std::max(mx,sd[3*sn+yy*N+xx]);} if(mx<=0.1f) for(int c=0;c<C;++c) sd[c*sn+y*N+x]=0.f; }
            std::copy(sd.begin(),sd.end(),st.begin()); }
        write_ppm(path.c_str(),st,0); };

    for(int it=1; it<=iters; ++it){
        // sample B pool slots; reseed the worst, damage the best couple
        std::vector<int> idx(B); for(int b=0;b<B;++b) idx[b]=(int)(randf()*POOL)%POOL;
        std::vector<std::pair<float,int>> rank; for(int b=0;b<B;++b) rank.push_back({rgba_mse(pool[idx[b]]),b});
        std::sort(rank.begin(),rank.end());
        seed_into(pool[idx[rank.back().second]]);                     // worst -> fresh seed
        if(B>=2) damage(pool[idx[rank[0].second]]);                   // best -> damage (learn to heal)
        if(B>=4) damage(pool[idx[rank[1].second]]);

        Tensor s=Tensor::zeros({B,C,N,N},false);
        for(int b=0;b<B;++b) std::copy(pool[idx[b]].begin(),pool[idx[b]].end(), s.data().begin()+(size_t)b*cn);

        int T=TMIN+(int)(randf()*(TMAX-TMIN+1));
        for(auto& p:params) p.zero_grad();
        for(int step=0; step<T; ++step){
            Tensor pc=conv2d(s,Wp,bp,1,1);
            Tensor h=relu(conv2d(pc,W1,b1,1,0));
            Tensor dx=conv2d(h,W2,b2,1,0);
            Tensor sm=Tensor::zeros({B,C,N,N},false);
            for(int b=0;b<B;++b) for(int i=0;i<sn;++i){ float u=randf()<0.5f?1.f:0.f; for(int c=0;c<C;++c) sm.data()[(b*C+c)*sn+i]=u; }
            s=add(s,mul(dx,sm));
            Tensor am=Tensor::zeros({B,C,N,N},false); const auto& sd=s.data();
            for(int b=0;b<B;++b) for(int y=0;y<N;++y) for(int x=0;x<N;++x){ float mx=0.f;
                for(int dy=-1;dy<=1;++dy)for(int dx2=-1;dx2<=1;++dx2){int yy=y+dy,xx=x+dx2; if(yy<0||yy>=N||xx<0||xx>=N)continue; mx=std::max(mx,sd[(b*C+3)*sn+yy*N+xx]);}
                float a=mx>0.1f?1.f:0.f; for(int c=0;c<C;++c) am.data()[(b*C+c)*sn+y*N+x]=a; }
            s=mul(s,am);
        }
        Tensor diff=mul(sub(s,tgt),chm);
        Tensor loss=mean(mul(diff,diff));
        loss.backward();
        ++at;
        for(size_t pi=0;pi<params.size();++pi){ auto& pd=params[pi].data(); auto& pg=params[pi].grad();
            for(size_t j=0;j<pd.size();++j){ float g=pg[j];
                mA[pi][j]=b1a*mA[pi][j]+(1-b1a)*g; vA[pi][j]=b2a*vA[pi][j]+(1-b2a)*g*g;
                float mh=mA[pi][j]/(1-std::pow(b1a,at)), vh=vA[pi][j]/(1-std::pow(b2a,at));
                pd[j]-=lr*mh/(std::sqrt(vh)+eps); } }
        // write evolved states back to the pool
        for(int b=0;b<B;++b) std::copy(s.data().begin()+(size_t)b*cn, s.data().begin()+(size_t)(b+1)*cn, pool[idx[b]].begin());
        if(it%25==0||it==1){ std::printf("iter %4d | T=%d | loss %.5f\n",it,T,loss.item()); std::fflush(stdout); }
        if(it%200==0){ save_weights(out); preview("preview.ppm"); }   // checkpoint + preview
    }

    // save weights (numel + floats each)
    { FILE* f=std::fopen(out.c_str(),"wb");
      if(f){ int meta[3]={N,C,H}; std::fwrite(meta,4,3,f);
        for(auto& t:std::vector<Tensor>{W1,b1,W2,b2}){ int n=t.numel(); std::fwrite(&n,4,1,f); std::fwrite(t.data().data(),4,n,f);} std::fclose(f);
        std::printf("saved %s (N=%d C=%d H=%d)\n",out.c_str(),N,C,H); } }

    // demo: grow, then damage, then keep running -> should reform (self-heal)
    std::vector<float> st(cn,0.f); seed_into(st);
    Tensor s=Tensor::zeros({1,C,N,N},false);
    auto run=[&](int steps){ for(int step=0;step<steps;++step){
        std::copy(st.begin(),st.end(),s.data().begin());
        Tensor pc=conv2d(s,Wp,bp,1,1); Tensor h=relu(conv2d(pc,W1,b1,1,0)); Tensor dx=conv2d(h,W2,b2,1,0);
        auto& sd=s.data(); for(int i=0;i<sn;++i){ float u=randf()<0.5f?1.f:0.f; for(int c=0;c<C;++c) sd[c*sn+i]+= (u?dx.data()[c*sn+i]:0.f); }
        for(int y=0;y<N;++y)for(int x=0;x<N;++x){ float mx=0.f; for(int dy=-1;dy<=1;++dy)for(int dx2=-1;dx2<=1;++dx2){int yy=y+dy,xx=x+dx2; if(yy<0||yy>=N||xx<0||xx>=N)continue; mx=std::max(mx,sd[3*sn+yy*N+xx]);} if(mx<=0.1f) for(int c=0;c<C;++c) sd[c*sn+y*N+x]=0.f; }
        std::copy(sd.begin(),sd.end(),st.begin()); } };
    run(TMAX); std::printf("\n== grown ==\n"); ascii_alpha(st,0); write_ppm("grown.ppm",st,0);
    damage(st); std::printf("\n== damaged ==\n"); ascii_alpha(st,0);
    run(TMAX); std::printf("\n== healed ==\n"); ascii_alpha(st,0); write_ppm("healed.ppm",st,0);
    std::printf("wrote grown.ppm / healed.ppm\n");
    return 0;
}
