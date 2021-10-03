
def get_ration(vols, hos, k):
    min_dis = []
    for h in hos:
        diss = [(h[0]-v[0])**2+(h[1]-v[1])**2 for v in vols]
        min_dis.append(min(diss))
    min_dis.sort()
    return min_dis[k-1]



if __name__ == '__main__':
    n, m, k = list(map(int, input().split()))
    v_c = []
    h_c = []
    for i in range(n):
        a, b = input().split()
        v_c.append((int(a), int(b)))
    for j in range(m):
        c, d = input().split()
        h_c.append((int(c), int(d)))
    print(get_ration(v_c, h_c, k))