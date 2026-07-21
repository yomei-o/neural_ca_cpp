import sys, zlib, struct
def ppm2png(src,dst):
    d=open(src,'rb').read()
    assert d[:2]==b'P6'
    # parse header
    idx=2; vals=[]
    while len(vals)<3:
        while idx<len(d) and d[idx] in b' \t\n\r': idx+=1
        if d[idx:idx+1]==b'#':
            while d[idx] not in b'\n': idx+=1
            continue
        j=idx
        while d[j] not in b' \t\n\r': j+=1
        vals.append(int(d[idx:j])); idx=j
    w,h,mx=vals; idx+=1
    raw=d[idx:idx+w*h*3]
    # PNG
    def chunk(t,data): return struct.pack('>I',len(data))+t+data+struct.pack('>I',zlib.crc32(t+data)&0xffffffff)
    out=b'\x89PNG\r\n\x1a\n'
    out+=chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))
    rows=b''.join(b'\x00'+raw[y*w*3:(y+1)*w*3] for y in range(h))
    out+=chunk(b'IDAT',zlib.compress(rows,9))
    out+=chunk(b'IEND',b'')
    open(dst,'wb').write(out)
ppm2png(sys.argv[1],sys.argv[2]); print("wrote",sys.argv[2])
