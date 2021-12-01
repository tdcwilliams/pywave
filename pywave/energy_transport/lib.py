import numpy as np


def diffl(u):
    d = np.zeros_like(u)
    d[1:] = np.diff(u)
    return d

    
def diffr(u):
    d = np.zeros_like(u)
    d[:-1] = np.diff(u)
    return d


def suml(u):
    s = np.zeros_like(u)
    s[0] = 2*u[0]
    s[1:] = u[1:] + u[:-1]
    return s

    
def sumr(u):
    s = np.zeros_like(u)
    s[:-1] = u[:-1] + u[1:]
    s[-1] = 2*u[-1]
    return s


def vector_min(u, v):
    """
    min function that works for vectors as well as floats
    
    Parameters:
    -----------
    u : float or numpy.ndarray
    v : float or numpy.ndarray
    
    Returns:
    --------
    min_uv : float or numpy.ndarray
    """
    return .5*(u+v - np.abs(u-v))


def vector_max(u, v):
    """
    max function that works for vectors as well as floats
    
    Parameters:
    -----------
    u : float or numpy.ndarray
    v : float or numpy.ndarray
    
    Returns:
    --------
    max_uv : float or numpy.ndarray
    """
    return .5*(u+v + np.abs(u-v))


def get_evec(lam, a, b, c, d):
    """
    get eigenvector for a 2x2 matrix for a given lambda
    - implemented to do many matrices at once

    Parameters:
    -----------
    lam : numpy.ndarray
        eigenvalue
    a: numpy.ndarray
        element (0,0) of source matrix
    b: numpy.ndarray
        element (0,1) of source matrix
    c: numpy.ndarray
        element (1,0) of source matrix
    d: numpy.ndarray
        element (1,1) of source matrix

    Returns:
    --------
    ev0 : numpy.ndarray
        1st element of normalised eigenvector
    ev1 : numpy.ndarray
        2nd element of normalised eigenvector
    """
    ev0 = np.zeros_like(lam)
    ev1 = np.zeros_like(lam)
    rows = [(a-lam,b), (c,d-lam)]
    hh = [np.hypot(*row) for row in rows]
    use0 = (hh[0] > hh[1])
    for (el0, el1), h, use in zip(rows, hh, (use0, ~use0)):
        if not np.any(use): continue
        ev0[use] = -el1[use]/h[use]
        ev1[use] = el0[use]/h[use]
    return ev0, ev1


def get_evec_aux(lam, a, b, c, d):
    """
    uv2 = exp(lam*t)*(aux_fac*ev*t + aux)
    WHERE IF ev0 != 0:
    auxiliary vector aux=(x,y) satisfies
    (a-lam)*x+b*y = aux_fac*ev0
    or
    h*aux.dot(aux) = h = aux_fac*ev0
    with aux=(a-lam,b)/h a unit vector,
    so aux_fac = h/ev0.

    IF ev0 ==0, THEN ev1 != 0:
    auxiliary vector aux=(x,y) satisfies
    c*x+(d-lam)*y = aux_fac*ev1
    or
    h*aux.dot(aux) = h = ev1
    with aux=(c,d-lam)/h a unit vector,
    so aux_fac = h/ev1.

    IN GENERAL:
    h = aux_fac * ev_

    NB aux' = aux +beta*(ev0,ev1) also works,
    but choose beta=0 so aux is orthogonal to (ev0,ev1).
    Then its coefficient is just is just (u0,v0).dot(aux)
    since aux is a unit vector

    Parameters:
    -----------
    lam : numpy.ndarray
        eigenvalue
    a: numpy.ndarray
        element (0,0) of source matrix
    b: numpy.ndarray
        element (0,1) of source matrix
    c: numpy.ndarray
        element (1,0) of source matrix
    d: numpy.ndarray
        element (1,1) of source matrix

    Returns:
    --------
    ev0 : numpy.ndarray
        1st element of normalised eigenvector
    ev1 : numpy.ndarray
        2nd element of normalised eigenvector
    aux0 : numpy.ndarray
        1st element of normalised  auxiliary vector
    aux1 : numpy.ndarray
        2nd element of normalised  auxiliary vector
    aux_fac: numpy.ndarray
        2nd solution is (aux_fac*t*ev + aux)*exp(lam*t)
    """
    ev0, ev1 = get_evec(lam, a, b, c, d)
    aux0 = np.zeros_like(lam)
    aux1 = np.zeros_like(lam)
    rhs = np.zeros_like(lam)
    rows = [(a-lam,b), (c,d-lam)]
    use0 = (np.abs(ev0) > np.abs(ev1))
    for (el0, el1), ev_, use in zip(rows, (ev0, ev1), (use0, ~use0)):
        if not np.any(use): continue
        aux0[use] = el0[use]
        aux1[use] = el1[use]
        rhs[use] = ev_[use]
    h = np.hypot(aux0, aux1)
    return ev0, ev1, aux0/h, aux1/h, h/rhs


def evolve_unique_evals(u, v, t, lams, abcd):
    """
    Evolve ODE
    [du/dt, dv/dt] = [[a,b],[c,d]](u,v)
    from time 0 to values in t
    when the eigenvalues of the matrix are not repeated.
    Note that in this case we don't need to know the 2nd
    row of the matrix (ie just need (a,b)).

    Parameters:
    -----------
    u : numpy.ndarray
        quantity to be advected to right
    v : numpy.ndarray
        quantity to be advected to left
    lams : list(numpy.ndarray)
        list of eigenvalues [lam1,lam2]
    abcd : tuple
        (a,b, c, d) with
            a: numpy.ndarray
                element (0,0) of source matrix
            b: numpy.ndarray
                element (0,1) of source matrix
            c: numpy.ndarray
                element (1,0) of source matrix
            d: numpy.ndarray
                element (1,1) of source matrix

    Returns:
    --------
    u_new : numpy.ndarray
        updated quantity to be advected to right
    v_new : numpy.ndarray
        updated quantity to be advected to left
    """
    if isinstance(t, np.ndarray):
        t_ = t.reshape(-1,1) # nt x 1
    else:
        t_ = np.array([[t]]) # nt x 1
    shp = (len(t_),len(u))
    u_new = np.zeros(shp)
    v_new = np.zeros(shp)
    for lam in lams:
        ev0, ev1 = get_evec(lam, *abcd)
        coef = ev0*u + ev1*v
        ex = np.exp(t_.dot(lam.reshape(1,-1))) #nt x nx
        u_new += ex * np.diag(coef*ev0)
        v_new += ex * np.diag(coef*ev1)
    return u_new, v_new


def evolve_repeated_eval(u, v, t, lam, abcd):
    """
    Evolve ODE
    [du/dt, dv/dt] = [[a,b],[c,d]](u,v)
    from time 0 to values in t
    when the eigenvalues of the matrix are repeated

    Parameters:
    -----------
    u : numpy.ndarray
        quantity to be advected to right
    v : numpy.ndarray
        quantity to be advected to left
    lam : numpy.ndarray
        eigenvalue
    abcd : tuple
        (a,b,c,d) with
        a: numpy.ndarray
            element (0,0) of source matrix
        b: numpy.ndarray
            element (0,1) of source matrix
        c: numpy.ndarray
            element (1,0) of source matrix
        d: numpy.ndarray
            element (1,1) of source matrix

    Returns:
    --------
    u_new : numpy.ndarray
        updated quantity to be advected to right
    v_new : numpy.ndarray
        updated quantity to be advected to left
    """

    t_ = t.reshape(-1,1)
    shp = (len(t),len(u))
    u_new = np.zeros(shp)
    v_new = np.zeros(shp)
    ev0, ev1, aux0, aux1, aux_fac = get_evec_aux(lam, *abcd)

    """ 1st solution: A*ev*exp(lam*t) """
    ex = np.exp(t.reshape(-1,1).dot(lam.reshape(1,-1)))
    coef = u*ev0 + v*ev1
    u_new = ex.dot(np.diag(coef * ev0))
    v_new = ex.dot(np.diag(coef * ev1))

    """
    2nd solution:
    u = B*(t*ev + aux)*exp(lam*t)
      = B*|aux|*(t*ev/|aux| + aux/|aux|)*exp(lam*t)
      = B'*(t*ev/h + aux')*exp(lam*t)
    where aux' is unit vector and h=|aux|.

    Below, aux is unit vector, and aux_fac=1/h.
    """
    coef = u*aux0 + v*aux1
    u_new += np.diag(t).dot(ex.dot(np.diag(coef*aux_fac*ev0)))
    v_new += np.diag(t).dot(ex.dot(np.diag(coef*aux_fac*ev1)))
    u_new += ex.dot(np.diag(coef*aux0))
    v_new += ex.dot(np.diag(coef*aux1))

    return u_new, v_new


def solve_2d_ode_spectral(u0, v0, t, a, b, c, d):
    """
    Evolve ODE
    [du/dt, dv/dt] = [[a,b],[c,d]](u,v)
    from time 0 to values in t

    Parameters:
    -----------
    u0 : numpy.ndarray
        initial values of u
    v0 : numpy.ndarray
        initial values of v
    t : numpy.ndarray
        time values to evaluate the solution
    a: numpy.ndarray
        element (0,0) of source matrix
    b: numpy.ndarray
        element (0,1) of source matrix
    c: numpy.ndarray
        element (1,0) of source matrix
    d: numpy.ndarray
        element (1,1) of source matrix

    Returns:
    --------
    u_new : numpy.ndarray
        u evaluated at all times
    v_new : numpy.ndarray
        v evaluated at all times
    """

    shp = (len(t),len(u0))
    u_all = np.zeros(shp)
    v_all = np.zeros(shp)
    discr = (a+d)**2 - 4*(a*d-b*c)
    assert(np.all(discr >= 0))
    lam_av = (a+d)/2 # average of the 2 eigenvalues

    # 1st do bits where matrix is diagonal
    # - u,v evolve independently
    diag = (b==0) * (c==0)
    use = np.copy(diag)
    if np.any(use):
        ex = np.exp(t.reshape(-1,1).dot(a[use].reshape(1,-1)))
        u_all[:,use] = ex.dot(u0[use].reshape(1,-1))
        ex = np.exp(t.reshape(-1,1).dot(d[use].reshape(1,-1)))
        v_all[:,use] = ex.dot(v0[use].reshape(1,-1))

    # 2nd do repeated eigenvalues (both = lam_av)
    # - NB matrix is not a multiple of the identity
    use = (discr == 0) * (not diag)
    if np.any(use):
        abcd = (a[use], b[use], c[use], d[use])
        u_all[:,use], v_all[:,use] = evolve_repeated_eval(
                u0[use], v0[use], t, lam_av[use], abcd)

    # 3rd do unique eigenvalues
    use = (discr > 0) * (not diag)
    if np.any(use):
        abcd = (a[use], b[use], c[use], d[use])
        dlam = np.sqrt(discr[use])/2
        lams = [lam_av[use] + dlam, lam_av[use]-dlam]
        u_all[:,use], v_all[:,use] = evolve_unique_evals(
                u0[use], v0[use], t, lams, abcd)

    return u_all, v_all
