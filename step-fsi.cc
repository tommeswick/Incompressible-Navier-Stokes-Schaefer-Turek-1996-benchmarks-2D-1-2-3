/**
 Thomas Wick 
 Institute of Applied Mathematics
 Leibniz University Hannover, Germany
 Date: May 13, 2016 (at RICAM Linz)
 E-mail: thomas.wick@ifam.uni-hannover.de

 Modified: Jan 29, 2020
           Sep 16, 2023
	   Sep 14, 2024
           Mar 12, 2025
	   Feb 04, 2026

 /// 
 Computes the Schaefer/Turek, 1996 benchmarck 2D-1,
 https://wwwold.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf
 with the benchmark results on p.12 and here in this code (they match perfectly):
 ------------------
 P-Diff:     1.1731485378884368e-01
 ------------------
 Face drag:  5.5589114445921277e+00
 Face lift:  1.0519679565305300e-02


 The 2D-2 and 2D-3 require updated 
 - inflow conditions, p.4
 - time intervals, p.4
 - and constants c_D and c_L in the drag and lift coefficients


 ///
 This code is a modification of 
 the ANS article open-source version:

 http://media.archnumsoft.org/10305/


 If you use the code or code-pieces, it would be very nice to cite 
 the above paper http://media.archnumsoft.org/10305/
 
 T. Wick; Solving Monolithic Fluid-Structure Interaction Problems
 in Arbitrary Lagrangian Eulerian Coordinates with the deal.II Library,
 Archive of Numerical Software, Vol. 1 (2013), pp. 1-19


 ///
 This code is based deal.II version 9.6.0

 
 ///
 This code is licensed under the GNU LESSER GENERAL PUBLIC LICENSE
 Version 2.1, February 1999

 Copyright 2012-2026: Thomas Wick 


*/



// Include files
//--------------

// The first step, as always, is to include
// the functionality of these 
// deal.II library files and some C++ header
// files.

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>  

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/tria_boundary_lib.h> // deprecated
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//#include <deal.II/lac/constraint_matrix.h>  // deprecated
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>


// C++
#include <fstream>
#include <sstream>

// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:				
using namespace dealii;




// First, we define tensors to access 
// certain variables in an easier way in 
// the assembling routines.  
namespace Tensors
{    
  template <int dim> 
    inline
    Tensor<2,dim> 
    get_pI (unsigned int q,
	    std::vector<Vector<double> > old_solution_values)
    {
      Tensor<2,dim> tmp;
      tmp[0][0] =  old_solution_values[q](dim);
      tmp[1][1] =  old_solution_values[q](dim);
      
      return tmp;      
    }

  template <int dim> 
    inline
    Tensor<2,dim> 
    get_pI_LinP (const double phi_i_p)
    {
      Tensor<2,dim> tmp;
      tmp.clear();
      tmp[0][0] = phi_i_p;    
      tmp[1][1] = phi_i_p;
      
      return tmp;
   }

  // pressure
 template <int dim> 
   inline
   Tensor<1,dim> 
   get_grad_p (unsigned int q,
	       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
   {     
     Tensor<1,dim> grad_p;     
     grad_p[0] =  old_solution_grads[q][dim][0];
     grad_p[1] =  old_solution_grads[q][dim][1];
      
     return grad_p;
   }

 template <int dim> 
  inline
  Tensor<1,dim> 
  get_grad_p_LinP (const Tensor<1,dim> phi_i_grad_p)	 
    {
      Tensor<1,dim> grad_p;      
      grad_p[0] =  phi_i_grad_p[0];
      grad_p[1] =  phi_i_grad_p[1];
	   
      return grad_p;
   }




  template <int dim> 
  inline
  Tensor<2,dim> 
  get_grad_v (unsigned int q,
	      std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
    {      
      Tensor<2,dim> grad_v;      
      grad_v[0][0] =  old_solution_grads[q][0][0];
      grad_v[0][1] =  old_solution_grads[q][0][1];
      grad_v[1][0] =  old_solution_grads[q][1][0];
      grad_v[1][1] =  old_solution_grads[q][1][1];
      
      return grad_v;
   }

  template <int dim> 
    inline
    Tensor<2,dim> 
    get_grad_v_T (const Tensor<2,dim> tensor_grad_v)
    {   
      Tensor<2,dim> grad_v_T;
      grad_v_T = transpose (tensor_grad_v);
            
      return grad_v_T;      
    }
  
  template <int dim> 
    inline
    Tensor<2,dim> 
    get_grad_v_LinV (const Tensor<2,dim> phi_i_grads_v)	 
    {     
        Tensor<2,dim> tmp;		 
	tmp[0][0] = phi_i_grads_v[0][0];
	tmp[0][1] = phi_i_grads_v[0][1];
	tmp[1][0] = phi_i_grads_v[1][0];
	tmp[1][1] = phi_i_grads_v[1][1];
      
	return tmp;
    }

  template <int dim> 
    inline
    Tensor<2,dim> 
    get_Identity ()
    {   
      Tensor<2,dim> identity;
      identity[0][0] = 1.0;
      identity[0][1] = 0.0;
      identity[1][0] = 0.0;
      identity[1][1] = 1.0;
            
      return identity;      
   }



 template <int dim> 
 inline
 Tensor<1,dim> 
 get_v (unsigned int q,
	std::vector<Vector<double> > old_solution_values)
    {
      Tensor<1,dim> v;	    
      v[0] = old_solution_values[q](0);
      v[1] = old_solution_values[q](1);
      
      return v;    
   }

 template <int dim> 
   inline
   Tensor<1,dim> 
   get_v_LinV (const Tensor<1,dim> phi_i_v)
   {
     Tensor<1,dim> tmp;
     tmp[0] = phi_i_v[0];
     tmp[1] = phi_i_v[1];
     
     return tmp;    
   }



} // end namespace tensors


// Class for initial condition values
  template <int dim>
  class InitialValues : public Function<dim>
  {
    public:
      InitialValues () : Function<dim>(dim+1) {}

      virtual double value (const Point<dim>   &p,
			    const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
				 Vector<double>   &value) const;

  };


  template <int dim>
  double
  InitialValues<dim>::value (const Point<dim>  &/*p*/,
			     const unsigned int component) const
  {
    // Anordnung Komponenten:
    // U=(vx,vy,p) = (0,1,2)
    // Only pressure
    if (component == 2)   
      {
	return 0.0; // Initial pressure if there is a wish: e.g., 2.6e+7;
      }
    
    return 0.0;
  }


  template <int dim>
  void
  InitialValues<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const
  {
    for (unsigned int comp=0; comp<this->n_components; ++comp)
      values (comp) = InitialValues<dim>::value (p, comp);
  }


//////////////////////////////////////////////////////////////////////////
template <int dim>
class NonhomDirichletBoundaryValues : public Function<dim> 
{
  public:
  NonhomDirichletBoundaryValues (const double time, 
				 const std::string test_case)    
    : Function<dim>(dim+1) 
    {
      _time = time;  
      _test_case = test_case;
    }
    
  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

  virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;

private:
  double _time;
  std::string _test_case;

};

// The boundary values are given to component 
// with number 1.
template <int dim>
double
NonhomDirichletBoundaryValues<dim>::value (const Point<dim>  &p,
			     const unsigned int component) const
{
  Assert (component < this->n_components,
	  ExcIndexRange (component, 0, this->n_components));


   const double long pi = 3.141592653589793238462643;

   // Copied from DOpElib (Feb 4, 2026)
   // # 2D-1: 0.3; 2D-2 and 2D-3: 1.5 
   double inflow_velocity = 0.3;
   
   if (_test_case == "2D-1_NSE" ||
	    _test_case == "2D-1_IV")
     {

       /*
      // Copied from DOpElib (Feb 4, 2026)
      # 2D-1: 0.3; 2D-2 and 2D-3: 1.5 
      set mean_inflow_velocity = 1.5
      //
      // Aufpassen bei der mean_inflow velocity und inflow_velocity, 
      // da unten noch mit 1,5 multipliziert wird, daher einmal
      // 0.3 und einmal 0.2 ... 
      // Das liegt an den FSI benchmarks, wo der Inflow leicht anders
      // mathematisch aufgeschrieben wurde. Es kommt aber dasselbe heraus.
      // 
      // Benchmark: BFAC 2D-1, 2D-2
      return ( (p(0) == 0) && (p(1) <= 0.41) ? -mean_inflow_velocity_ *
         (4.0/0.1681) *
         (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
      
      // Benchmark: BFAC 2D-3
      return ( (p(0) == 0) && (p(1) <= 0.41) ? -mean_inflow_velocity_ *
               (4.0/0.1681) *
               std::sin(M_PI * mytime/8) *
               (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
       */

       
       if (component == 0)   
	 {
	   // Smooth inflow profile for better initial start. 
	   if (_time < 2.0)
	     {
	       return   ( (p(0) == 0) && (p(1) <= 0.41) ? -/*1.5 */ inflow_velocity * 
			  (1.0 - std::cos(pi/2.0 * _time))/2.0 * 
			  (4.0/0.1681) *
			  // 2D-3 (ansonsten bei 2D-1, 2D-2 auskommentieren)
			  //std::sin(pi * _time/8.0) *
			  (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
	     }
	   else 
	     {
	       return ( (p(0) == 0) && (p(1) <= 0.41) ? -/*1.5 */ inflow_velocity * 			
			(4.0/0.1681) *
			// 2D-3 (ansonsten bei 2D-1, 2D-2 auskommentieren)
			//std::sin(pi * _time/8.0) *
			(std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
	       
	     }
	   
	 }
     }
   else 
     {
       std::cout << "Aborting. In NonhomDirichletBoundaryValues." << std::endl;
       abort ();
     }

 
  return 0;
}



template <int dim>
void
NonhomDirichletBoundaryValues<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = NonhomDirichletBoundaryValues<dim>::value (p, c);
}







//////////////////////////////////////////////////////////////////////////
// In the next class, we define the main problem at hand.

// The  program is organized as follows. First, we set up
// runtime parameters and the system as done in other deal.II tutorial steps. 
// Then, we assemble
// the system matrix (Jacobian of Newton's method) 
// and system right hand side (residual of Newton's method) for the non-linear
// system. Two functions for the boundary values are provided because
// we are only supposed to apply boundary values in the first Newton step. In the
// subsequent Newton steps all Dirichlet values have to be equal zero.
// Afterwards, the routines for solving the linear 
// system and the Newton iteration are self-explaining. The following
// function is standard in deal.II tutorial steps:
// writing the solutions to graphical output. 
// The last three functions provide the framework to compute 
// functional values of interest.     
template <int dim>
class StepFlowNSE 
{
public:
  
  StepFlowNSE (const unsigned int degree);
  ~StepFlowNSE (); 
  void run ();
  
private:
  
  // Setup of material parameters, time-stepping scheme
  // spatial grid, etc.
  void set_runtime_parameters ();

  // Create system matrix, rhs and distribute degrees of freedom.
  void setup_system ();

  // Assemble left and right hand side for Newton's method
  void assemble_system_matrix ();   
  void assemble_system_rhs ();
  
  // Boundary conditions (bc)
  void set_bc_in_initial_newton_guess ();
  void set_bc_in_subsequent_newton_iter ();
  
  // Linear solver
  void solve ();

  // Nonlinear solver
  void newton_iteration();

  // Graphical visualization of output			  
  void output_results (const unsigned int refinement_cycle,
		       const BlockVector<double> solution_1) const;

  // Evaluation of functional values   
  double compute_point_value (Point<dim> p,
			      const unsigned int component) const;

  void compute_drag_lift_tensor (); 
  void compute_functional_values (); 


  const unsigned int   degree;
  
  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  //ConstraintMatrix     constraints;
  AffineConstraints<double> constraints;
  
  BlockSparsityPattern      sparsity_pattern; 
  BlockSparseMatrix<double> system_matrix; 
  
  BlockVector<double> solution, newton_update, old_timestep_solution, old_old_timestep_solution;
  BlockVector<double> system_rhs;

  TimerOutput         timer;
  
  // Global variables for timestepping scheme   
  unsigned int timestep_number;
  unsigned int max_no_timesteps;  
  double timestep, theta, time; 
  std::string time_stepping_scheme;

  double force_fluid_x, force_fluid_y;	  
  

  double volume_source, traction_x, traction_y, traction_x_biot, traction_y_biot;
  
  // Fluid parameters
  double density_fluid, kinematic_viscosity;

  SparseDirectUMFPACK A_direct;

  std::string test_case;

  std::string filename_basis;


};


// The constructor of this class is comparable 
// to other tutorials steps, e.g., step-22, and step-31. 
// We are going to use the following finite element discretization: 
// Q_2^c for the velocities,
// Q_1^c for the pressure.
// This combination, specifically for the NSE part, is inf-sup stable. 
template <int dim>
StepFlowNSE<dim>::StepFlowNSE (const unsigned int degree)
                :
                degree (degree),
		triangulation (Triangulation<dim>::maximum_smoothing),
                fe (FE_Q<dim>(degree+1), dim,     // velocity (vector-valued)                 
		    FE_Q<dim>(degree),   1),      // pressure (single-valued)
                dof_handler (triangulation),
		timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)		
{}


// This is the standard destructor.
template <int dim>
StepFlowNSE<dim>::~StepFlowNSE () 
{}



// In this method, we set up runtime parameters that 
// could also come from a paramter file. 
template <int dim>
void StepFlowNSE<dim>::set_runtime_parameters ()
{

  std::string grid_name;

  if (test_case == "2D-1_NSE")
    {
      // Fluid parameters for all test cases
      density_fluid           = 1.0;
      kinematic_viscosity     = 1.0e-3; 

      // Timestepping schemes
      //BE, CN, CN_shifted
      // 2D-1: BE
      // 2D-2, 2D-3: CN or CN_shifted
      time_stepping_scheme = "BE";

      // Timestep size:
      // 2D-1: 1.0; 2D-2: 0.1, 2D-3: 0.1
      timestep = 1.0; 
      
      // Max number of time steps
      // 2D-1: 25 ; 2D-2: 80; 2D-3: 80
      max_no_timesteps = 25; 

      grid_name  = "nsbench4_original.inp";
      //grid_name = "rectangle_mandel.inp";
      //grid_name = "cylinder-structured.msh";
      
      GridIn<dim> grid_in;
      grid_in.attach_triangulation (triangulation);
      std::ifstream input_file(grid_name.c_str());      
      Assert (dim==2, ExcInternalError());
      grid_in.read_ucd (input_file);
      //grid_in.read_msh(input_file);

      
      // TODO: only when ns bench 2D-1, 2D-2, 2D-3
      Point<dim> p(0.2, 0.2);
      //double radius = 0.05;
      const SphericalManifold<dim> boundary(p);
      triangulation.set_all_manifold_ids_on_boundary(80,8);
      //triangulation.set_all_manifold_ids_on_boundary(81,9);
      triangulation.set_manifold (8, boundary);
      //triangulation.set_manifold (9, boundary);
      
      
      triangulation.refine_global (4); 

      filename_basis  = "solution_2D_1_NSE_"; 

    }
  else
    {
      std::cout << "Aborting in set runtime parameters." << std::endl;
      abort();
    }


  // Right hand side flow equation, e.g., gravity
  force_fluid_x = 0.0;
  force_fluid_y = 0.0;

  // Traction - not in use here
  traction_x = 0.0;
  traction_y = 0.0;

  
  // A variable to count the number of time steps
  timestep_number = 0;

  // Counts total time  
  time = 0;
 
  // Here, we choose a time-stepping scheme that
  // is based on finite differences:
  // BE         = backward Euler scheme 
  // CN         = Crank-Nicolson scheme
  // CN_shifted = time-shifted Crank-Nicolson scheme 
  // For further properties of these schemes,
  // we refer to standard literature.
  if (time_stepping_scheme == "BE")
    theta = 1.0;
  else if (time_stepping_scheme == "CN")
    theta = 0.5;
  else if (time_stepping_scheme == "CN_shifted")
    theta = 0.5 + 0.1; //timestep;
  else 
    std::cout << "No such timestepping scheme" << std::endl;

}



// This function is similar to many deal.II tuturial steps.
template <int dim>
void StepFlowNSE<dim>::setup_system ()
{
  timer.enter_subsection("Setup system.");

  system_matrix.clear ();
  
  dof_handler.distribute_dofs (fe);  
  DoFRenumbering::Cuthill_McKee (dof_handler);

  // We are dealing with 5 components for this 
  // two-dimensional Biot problem
  // (the first component is actually unnecessary 
  // and a relict from the original fluid-structure 
  // interaction code. But this component 
  // might be used for future extensions with other equations.
  // velocity in x and y:                0
  // scalar pressure field:              1
  std::vector<unsigned int> block_component (3,0);
  block_component[dim] = 1;
 
  DoFRenumbering::component_wise (dof_handler, block_component);

  {				 
    constraints.clear ();
    set_bc_in_subsequent_newton_iter ();
    DoFTools::make_hanging_node_constraints (dof_handler,
					     constraints);
  }
  constraints.close ();
  
  //std::vector<unsigned int> dofs_per_block (3);
  std::vector<types::global_dof_index> dofs_per_block (2);
  dofs_per_block = DoFTools::count_dofs_per_fe_block (dof_handler, block_component);  
  const unsigned int n_v = dofs_per_block[0],
    n_p =  dofs_per_block[1]; // pressure

  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl  	  
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << " (" << n_v << '+' << n_p <<  ')'
            << std::endl;


 
      
 {
    BlockDynamicSparsityPattern csp (2,2);

    csp.block(0,0).reinit (n_v, n_v);
    csp.block(0,1).reinit (n_v, n_p);
  
    csp.block(1,0).reinit (n_p, n_v);
    csp.block(1,1).reinit (n_p, n_p);
  
    csp.collect_sizes();    
  

    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);

    sparsity_pattern.copy_from (csp);
  }
 
 system_matrix.reinit (sparsity_pattern);

  // Actual solution at time step n
  solution.reinit (2);
  solution.block(0).reinit (n_v);
  solution.block(1).reinit (n_p);
 
  solution.collect_sizes ();
 
  // Old timestep solution at time step n-1
  old_timestep_solution.reinit (2);
  old_timestep_solution.block(0).reinit (n_v);
  old_timestep_solution.block(1).reinit (n_p);
 
  old_timestep_solution.collect_sizes ();

  // Updates for Newton's method
  newton_update.reinit (2);
  newton_update.block(0).reinit (n_v);
  newton_update.block(1).reinit (n_p);
 
  newton_update.collect_sizes ();
 
  // Residual for  Newton's method
  system_rhs.reinit (2);
  system_rhs.block(0).reinit (n_v);
  system_rhs.block(1).reinit (n_p);

  system_rhs.collect_sizes ();
  
  timer.leave_subsection(); 
}


// In this function, we assemble the Jacobian matrix
// for the Newton iteration. 
//
// Assembling of the inner most loop is treated with help of 
// the fe.system_to_component_index(j).first function from
// the library. 
// Using this function makes the assembling process much faster
// than running over all local degrees of freedom. 
template <int dim>
void StepFlowNSE<dim>::assemble_system_matrix ()
{  
  //std::cout << "Entering assemble matrix." << std::endl;
  Timer timer_assemble;
  timer_assemble.start();
  
  timer.enter_subsection("Assemble Matrix.");
  system_matrix=0;
     
  QGauss<dim>   quadrature_formula(degree+2);  
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);
  
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);
   
  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

  //std::vector<unsigned int> local_dof_indices (dofs_per_cell); 
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Now, we are going to use the 
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim); // 2

  // We declare Vectors and Tensors for 
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double> > old_solution_values (n_q_points, 
				 		    Vector<double>(dim+1));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								std::vector<Tensor<1,dim> > (dim+1));

  std::vector<Vector<double> >  old_solution_face_values (n_face_q_points, 
							  Vector<double>(dim+1));
       
  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, 
								     std::vector<Tensor<1,dim> > (dim+1));
    

  // We declare Vectors and Tensors for 
  // the solution at the previous time step:
  std::vector<Vector<double> > old_timestep_solution_values (n_q_points, 
				 		    Vector<double>(dim+1));


  std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points, 
  					  std::vector<Tensor<1,dim> > (dim+1));


  std::vector<Vector<double> >   old_timestep_solution_face_values (n_face_q_points, 
								    Vector<double>(dim+1));
  
    
  std::vector<std::vector<Tensor<1,dim> > >  old_timestep_solution_face_grads (n_face_q_points, 
									       std::vector<Tensor<1,dim> > (dim+1));
   
  // Declaring test functions:
  std::vector<Tensor<1,dim> > phi_i_v (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_v(dofs_per_cell);
  std::vector<double>         phi_i_p(dofs_per_cell); 
  std::vector<Tensor<1,dim> > phi_i_grads_p (dofs_per_cell);
 				     				   
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  

  unsigned int cell_counter = 0;
  for (; cell!=endc; ++cell)
    { 
      fe_values.reinit (cell);
      local_matrix = 0;
      
      
      // Old Newton iteration values
      fe_values.get_function_values (solution, old_solution_values);
      fe_values.get_function_gradients (solution, old_solution_grads);
      
      // Old_timestep_solution values
      fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
      // Material id = 0 
      if (cell->material_id() == 0)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	      
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_i_v[k]       = fe_values[velocities].value (k, q);
		  phi_i_grads_v[k] = fe_values[velocities].gradient (k, q);
		  phi_i_p[k]       = fe_values[pressure].value (k, q);	
		  phi_i_grads_p[k] = fe_values[pressure].gradient (k, q);
		}
	      
	      const Tensor<1,dim> v = Tensors
		::get_v<dim> (q, old_solution_values);

	      const Tensor<2,dim> grad_v = Tensors 
		::get_grad_v<dim> (q, old_solution_grads); 

	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  
		  Tensor<2, 2> pI_LinP;
		  pI_LinP.clear();
		  pI_LinP[0][0] = phi_i_p[i];
		  pI_LinP[1][1] = phi_i_p[i];

		  // Stress tensor flow (Newtonian fluid)
		  const Tensor<2,dim>  stress_fluid_LinAll = 
		    density_fluid * kinematic_viscosity * (phi_i_grads_v[i] + transpose(phi_i_grads_v[i]));		   
		    
		  const Tensor<1,dim> convection_fluid_LinAll = density_fluid * (phi_i_grads_v[i] * v + grad_v * phi_i_v[i]) ;
		    
		  const double incompressibility_LinAll = phi_i_grads_v[i][0][0] + phi_i_grads_v[i][1][1];
		      
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      
		      const unsigned int comp_j = fe.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1)
			{			  
			  // Compute velocities v of Navier-Stokes flow
			  local_matrix(j,i) += (density_fluid * phi_i_v[i] * phi_i_v[j] 
						+ timestep * theta * convection_fluid_LinAll * phi_i_v[j]
						// Pressure stress
						+ timestep * scalar_product(-pI_LinP , phi_i_grads_v[j])						
						// Velocities stress					       
						+ timestep * theta * scalar_product(stress_fluid_LinAll, phi_i_grads_v[j])
						) *  fe_values.JxW(q);			  
			}		     
		      else if (comp_j == 2)
			{
			  // Pressure equation: Newton linearization
			  local_matrix(j,i) += (incompressibility_LinAll *  phi_i_p[j] 
						) * fe_values.JxW(q);      
			}
		      // end j dofs
		    }  
		  // end i dofs		     
		}   
	      // end n_q_points 
	    }    
	  
	  cell->get_dof_indices (local_dof_indices);
	  constraints.distribute_local_to_global (local_matrix, local_dof_indices,
						  system_matrix);

	} 
      // Second material id
      else if (cell->material_id() == 1)
	{	  
	  // Currently not implemented in this code.
	  std::cout << "Material id not implemented." << std::endl;
	  abort();
	  
	} 
      

      cell_counter++;
    
    }  // end cell
  
  timer.leave_subsection();

  // Output explicitly CPU time for assemble matrix
  //timer_assemble.stop();
  //std::cout << "Assemble matrix: " << timer_assemble.cpu_time() << std::endl;
  //timer_assemble.reset();
}



// In this function we assemble the semi-linear 
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the 
// system matrix.
template <int dim>
void
StepFlowNSE<dim>::assemble_system_rhs ()
{
  timer.enter_subsection("Assemble Rhs.");
  system_rhs=0;
  
  QGauss<dim>   quadrature_formula(degree+2);
  QGauss<dim-1> face_quadrature_formula(degree+2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);

  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
				    update_values         | update_quadrature_points  |
				    update_normal_vectors | update_gradients |
				    update_JxW_values);

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  
  const unsigned int   n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points   = face_quadrature_formula.size();
 
  Vector<double>       local_rhs (dofs_per_cell);

  //std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim); 
 
  std::vector<Vector<double> > 
    old_solution_values (n_q_points, Vector<double>(dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+1));


  std::vector<Vector<double> > 
    old_solution_face_values (n_face_q_points, Vector<double>(dim+1));
  
  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+1));
  
  std::vector<Vector<double> > 
    old_timestep_solution_values (n_q_points, Vector<double>(dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+1));

  std::vector<Vector<double> > 
    old_timestep_solution_face_values (n_face_q_points, Vector<double>(dim+1));
     
  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+1));


  std::vector<Vector<double> > 
    old_old_timestep_solution_values (n_q_points, Vector<double>(dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_old_timestep_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+1));
   

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  unsigned int cell_counter = 0;
  for (; cell!=endc; ++cell)
    { 
      fe_values.reinit (cell);	 
      local_rhs = 0;   	
       
      // old Newton iteration
      fe_values.get_function_values (solution, old_solution_values);
      fe_values.get_function_gradients (solution, old_solution_grads);
            
      // old timestep iteration
      fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
      //
      if (cell->material_id() == 0)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	
	      const Tensor<2,dim> pI = Tensors
		::get_pI<dim> (q, old_solution_values);

	      const Tensor<1,dim> v = Tensors
		::get_v<dim> (q, old_solution_values);

	      const Tensor<2,dim> grad_v = Tensors 
		::get_grad_v<dim> (q, old_solution_grads);
	      
	      // Stress tensor fluid
	      Tensor<2,dim> sigma_fluid;
	      sigma_fluid.clear();
	      sigma_fluid = density_fluid * kinematic_viscosity * (grad_v + transpose(grad_v));

	      // Convection term of the fluid 
	      Tensor<1,dim> convection_fluid;
	      convection_fluid.clear();
	      convection_fluid = density_fluid * grad_v * v;
	      
	      // Divergence of the fluid 
	      const double incompressiblity_fluid = grad_v[0][0] + grad_v[1][1];				  
	       
	      Tensor<1,dim> fluid_force;
	      fluid_force.clear();
	      fluid_force[0] = force_fluid_x;
	      fluid_force[1] = force_fluid_y;

	      // Old time step values
	      const Tensor<1,dim> old_timestep_v = Tensors
		::get_v<dim> (q, old_timestep_solution_values);

	      const Tensor<2,dim> old_timestep_grad_v = Tensors 
		::get_grad_v<dim> (q, old_timestep_solution_grads);

	      // Info: be careful: when force is time-dependent, then
	      // this must be taken from the previous time step!!!
	      Tensor<1,dim> old_timestep_fluid_force;
	      old_timestep_fluid_force.clear();
	      old_timestep_fluid_force[0] = force_fluid_x;
	      old_timestep_fluid_force[1] = force_fluid_y;

	      
	      Tensor<2,dim> old_timestep_sigma_fluid;
	      old_timestep_sigma_fluid.clear();
	      old_timestep_sigma_fluid = 		
		density_fluid * kinematic_viscosity  * (old_timestep_grad_v + transpose(old_timestep_grad_v));	      			
		

		
	      Tensor<1,dim> old_timestep_convection_fluid;
	      old_timestep_convection_fluid.clear();
	      old_timestep_convection_fluid = density_fluid * old_timestep_grad_v * old_timestep_v;
	      
	      	      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  const unsigned int comp_i = fe.system_to_component_index(i).first; 
		  if (comp_i == 0 || comp_i == 1)
		    { 
		      // Compute velocities of Navier-Stokes flow
		      const Tensor<1,dim> phi_i_v = fe_values[velocities].value (i, q);
		      const Tensor<2,dim> phi_i_grads_v = fe_values[velocities].gradient (i, q);

		      local_rhs(i) -=  (density_fluid * (v - old_timestep_v) * phi_i_v
					+ timestep * theta * convection_fluid * phi_i_v
					+ timestep * (1.0 - theta) * old_timestep_convection_fluid * phi_i_v
					+ timestep * scalar_product(-pI, phi_i_grads_v)
					+ timestep * theta * scalar_product(sigma_fluid, phi_i_grads_v)
					+ timestep * (1.0 - theta) * scalar_product(old_timestep_sigma_fluid, phi_i_grads_v)
					// Right hand side (e.g. gravitation)
				       - timestep * theta * fluid_force * phi_i_v   
				       - timestep * (1.0 - theta) * old_timestep_fluid_force * phi_i_v  
					) * fe_values.JxW(q);    

		      
		    }		
		  else if (comp_i == 2)
		    {		     
		      const double phi_i_p = fe_values[pressure].value (i, q);
		      local_rhs(i) -= (incompressiblity_fluid * phi_i_p
				       ) * fe_values.JxW(q);  
		      
		    }
		  // end i	  
		} 	
	      // end n_q_points 		   
	    } 
	  
	  /*
	  // Pressure Neumann conditions
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->face(face)->at_boundary() && 		  
		  (cell->face(face)->boundary_id() == 0) 
		  )
		{
		  
		  fe_face_values.reinit (cell, face);
		  
		  fe_face_values.get_function_values (solution, old_solution_face_values);
		  fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
		
	 
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {	
		      Tensor<1,dim> neumann_value;
		      neumann_value[0] = 1.0;
		      neumann_value[1] = 0.0;
			
		      double fluid_pressure = old_timestep_solution_face_values[q](dim);

		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  const unsigned int comp_i = fe.system_to_component_index(i).first; 
			  if (comp_i == 0 || comp_i == 1)
			    {  
			      local_rhs(i) +=  (timestep * theta * neumann_value * fe_face_values[velocities].value (i, q) 	     
						) * fe_face_values.JxW(q);					   
			    }
			  // end i
			}  
		      // end face_n_q_points    
		    }                                     
		} 
	    }  // end face integrals
	  */


	  cell->get_dof_indices (local_dof_indices);
	  constraints.distribute_local_to_global (local_rhs, local_dof_indices,
						  system_rhs);
	  
	// end if (for material id 0)  
	}   
      // Second material
      else if (cell->material_id() == 1)
	{
	  std::cout << "Material id not implemented." << std::endl;
	  abort();
	}   
      
      cell_counter++;

    }  // end cell
      
  timer.leave_subsection();
}


// Here, we impose boundary conditions
// for the whole system. 
// `Initial' means that we impose non-homoegeneous Dirichlet
//  condition on the `initial' guess of Newton's method. After,
// these must be set two homogeneous Dirichlet conditions, which
// is done in set_bc_in_subsequent_newton_iter ()
template <int dim>
void
StepFlowNSE<dim>::set_bc_in_initial_newton_guess ()
{ 
  // Boundary colors
  //
  //   ___3___
  //   |     |
  //  0|     |1 
  //   |_____|
  //      2

  //std::map<unsigned int,double> boundary_values;
    std::map<types::global_dof_index, double> boundary_values;
    std::vector<bool> component_mask (dim+1, true);
    // 0 = vx  // test function
    // 1 = vy  // test function
    // 2 = pressure

    // (Scalar) pressure
    component_mask[dim] = false; // false  
 
    component_mask[0]         = true;
    component_mask[1]         = true;
    VectorTools::interpolate_boundary_values (dof_handler,
					      0,
					      //ZeroFunction<dim>(dim+1),  
					      // Use this function for nonhomogeneous
					      // Dirichlet conditions
					      NonhomDirichletBoundaryValues<dim>(time,test_case),
					      boundary_values,
					      component_mask);  

    component_mask[0]       = false;
    component_mask[1]       = false;
    VectorTools::interpolate_boundary_values (dof_handler,
					      1,
					      dealii::Functions::ZeroFunction<dim>(dim+1),					
					      boundary_values,
					      component_mask);

 
    component_mask[0]       = true;
    component_mask[1]       = true;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
					      dealii::Functions::ZeroFunction<dim>(dim+1), 
					      boundary_values,
                                              component_mask);

    component_mask[0]         = true;
    component_mask[1]         = true;
    VectorTools::interpolate_boundary_values (dof_handler,
					      3,
					      dealii::Functions::ZeroFunction<dim>(dim+1),  
					      boundary_values,
					      component_mask);
    if (test_case == "2D-1_NSE" || 
	test_case == "2D-1_IV")
      { 
	VectorTools::interpolate_boundary_values (dof_handler,
						  80,
						  dealii::Functions::ZeroFunction<dim>(dim+1),  
						  boundary_values,
						  component_mask);
	
	VectorTools::interpolate_boundary_values (dof_handler,
						  81,
						  dealii::Functions::ZeroFunction<dim>(dim+1),  
						  boundary_values,
						  component_mask);
      }

    // Info: old implementation until deal.II 9.4.0
//    for (typename std::map<unsigned int, double>::const_iterator
//	   i = boundary_values.begin();
//	 i != boundary_values.end();
//	 ++i)
//      solution(i->first) = i->second;
    for (auto &boundary_value : boundary_values)
      solution(boundary_value.first) = boundary_value.second;
    
}

// This function applies boundary conditions 
// to the Newton iteration steps. For all variables that
// have non-homogeneous Dirichlet conditions on some (or all) parts
// of the outer boundary, we apply zero-Dirichlet
// conditions, now. 
template <int dim>
void
StepFlowNSE<dim>::set_bc_in_subsequent_newton_iter ()
{
  // Boundary colors
  //
  //   ___3___
  //   |     |
  //  0|     |1 
  //   |_____|
  //      2


    std::vector<bool> component_mask (dim+1, true);
    component_mask[dim]   = false; // temp  

    component_mask[0]         = true;
    component_mask[1]         = true;
    VectorTools::interpolate_boundary_values (dof_handler,
					      0,
					      dealii::Functions::ZeroFunction<dim>(dim+1), 
					      constraints,				
					      component_mask);  

    component_mask[0]       = false;
    component_mask[1]       = false;
    VectorTools::interpolate_boundary_values (dof_handler,
					      1,
					      dealii::Functions::ZeroFunction<dim>(dim+1),  
					      constraints,
					      component_mask);

 
    component_mask[0]       = true;
    component_mask[1]       = true;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              2,
					      dealii::Functions::ZeroFunction<dim>(dim+1),  
					      constraints,
                                              component_mask);

    
    
    component_mask[0]       = true;
    component_mask[1]       = true;
    VectorTools::interpolate_boundary_values (dof_handler,
					      3,
					      dealii::Functions::ZeroFunction<dim>(dim+1),  
					      constraints,		
					      component_mask);
    if (test_case == "2D-1_NSE" || 
	test_case == "2D-1_IV")
      {
	VectorTools::interpolate_boundary_values (dof_handler,
						  80,
						  dealii::Functions::ZeroFunction<dim>(dim+1),  
						  constraints,
						  component_mask);
	
	VectorTools::interpolate_boundary_values (dof_handler,
						  81,
						  dealii::Functions::ZeroFunction<dim>(dim+1),  
						  constraints,
						  component_mask);      
      }

    

}  

// In this function, we solve the linear systems
// inside the nonlinear Newton iteration. We just
// use a direct solver from UMFPACK.
template <int dim>
void 
StepFlowNSE<dim>::solve () 
{
  //std::cout << "Entering linear solver." << std::endl;
  Timer timer_linear_solve;
  timer_linear_solve.start();
  
  timer.enter_subsection("Solve linear system.");
  Vector<double> sol, rhs;    
  sol = newton_update;    
  rhs = system_rhs;
  
  //SparseDirectUMFPACK A_direct;
  //A_direct.factorize(system_matrix);     
  A_direct.vmult(sol,rhs); 
  newton_update = sol;
  
  constraints.distribute (newton_update);
  timer.leave_subsection();

  // Output explicitly the LU solve (not factorization !!!)
  //timer_linear_solve.stop();
  //std::cout << "Linear solve: " << timer_linear_solve.cpu_time() << std::endl;
  //timer_linear_solve.reset();
}

// This is the Newton iteration 
// to solve the non-linear system of equations:
//
// A'(U^k)(\delta U, \Psi) = - A(U^k)(\Psi)
//                 U^{k+1} = U^k + \lambda \delta U
//
// with $\lambda\in (0,1]$.
//
// For Newton's method as implemented here, please see 
// for instance Chapter 13 in https://doi.org/10.15488/9248
//
// A'(U^k)(\delta U, \Psi) ^= assemble_system_matrix ()
// A(U^k)(\Psi)            ^= assemble_system_rhs ()
//
// The Jacobian A'(U^k)(\delta U, \Psi) is obtained
// by calculating the directional derivatives.
// The minus sign before the residual A(U^k)(\Psi) 
// is contained in local_rhs(i) -= ... where we use
//   -= rather than +=
//  
//
template <int dim>
void StepFlowNSE<dim>::newton_iteration () 
{ 
  Timer timer_newton;
  const double lower_bound_newton_residual = 1.0e-10; 
  const unsigned int max_no_newton_steps  = 20;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 1e-10; //0.1; 
 
  // Line search parameters
  unsigned int line_search_step;
  const unsigned int  max_no_line_search_steps = 10;
  const double line_search_damping = 0.6;
  double new_newton_residual;
  
  // Application of the initial boundary conditions to the 
  // variational equations:
  set_bc_in_initial_newton_guess ();
  assemble_system_rhs();

  double newton_residual = system_rhs.l2_norm(); 
  double old_newton_residual = newton_residual;
  double initial_newton_residual = newton_residual;
  unsigned int newton_step = 1;
   
  unsigned int stop_when_line_search_two_times_max_number = 0;


  if (newton_residual < lower_bound_newton_residual)
    {
      std::cout << '\t' 
		<< std::scientific 
		<< newton_residual 
		<< std::endl;     
    }

  // Newton stopping criterion: combination of absolute and
  // relative residuals
  while ((newton_residual > lower_bound_newton_residual &&
	  (newton_residual/initial_newton_residual) > lower_bound_newton_residual) &&
	 newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residual = newton_residual;
      
      assemble_system_rhs();
      newton_residual = system_rhs.l2_norm();

      if (newton_residual < lower_bound_newton_residual)
	{
	  std::cout << '\t' 
		    << std::scientific 
		    << newton_residual << std::endl;
	  break;
	}
  
      // Check if matrix needs to re-build. If not 
      // then we save some time and perform quasi-Newton steps.
      if (newton_residual/old_newton_residual > nonlinear_rho)
	{
	  assemble_system_matrix ();	

	  // Terminal output of CPU timings
	  //std::cout << "Entering linear factorization." << std::endl;
	  //Timer timer_linear_fact;
	  //timer_linear_fact.start();

	  // Only factorize when matrix is re-built (this is the expensive step in the linear solution)
	  A_direct.factorize(system_matrix);

	  // Terminal output of CPU timings (cont'd)
	  //timer_linear_fact.stop();
	  //std::cout << "Linear factorization: " << timer_linear_fact.cpu_time() << std::endl;
	  //timer_linear_fact.reset();
	}   

      // Solve Ax = b
      solve ();	  
        
      line_search_step = 0;	  
      for ( ; 
	    line_search_step < max_no_line_search_steps; 
	    ++line_search_step)
	{	     					 
	  solution += newton_update;
	  
	  assemble_system_rhs ();			
	  new_newton_residual = system_rhs.l2_norm();
	  
	  if (new_newton_residual < newton_residual)
	      break;
	  else 	  
	    solution -= newton_update;
	  
	  newton_update *= line_search_damping;
	}	   
     
      if (line_search_step == 10)
	stop_when_line_search_two_times_max_number ++;

      if (stop_when_line_search_two_times_max_number == 3)
	{
	  std::cout << "Aborting Newton as line search does not help to converge anymore." << std::endl;
	  abort();
	}

      timer_newton.stop();
      
      std::cout << std::setprecision(5) <<newton_step << '\t' 
		<< std::scientific << newton_residual << '\t'
		<< std::scientific << newton_residual/initial_newton_residual << '\t'
		<< std::scientific << newton_residual/old_newton_residual  <<'\t' ;
      if (newton_residual/old_newton_residual > nonlinear_rho)
	std::cout << "r" << '\t' ;
      else 
	std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t' 
		<< std::scientific << timer_newton.cpu_time ()
		<< std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;      
    }
}


// This function is known from almost all other 
// tutorial steps in deal.II.
template <int dim>
void
StepFlowNSE<dim>::output_results (const unsigned int refinement_cycle,
							const BlockVector<double> output_vector_1)  const
{

  std::vector<std::string> solution_names_1;
  solution_names_1.push_back ("x_velo");
  solution_names_1.push_back ("y_velo"); 
  solution_names_1.push_back ("press");

   
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation_1
    (dim+1, DataComponentInterpretation::component_is_scalar);


  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);  
   
  data_out.add_data_vector (output_vector_1, solution_names_1,
			    DataOut<dim>::type_dof_data,			   
			    data_component_interpretation_1);
  

  // INFO: Build higher order patches
  data_out.build_patches (2);



   
  std::ostringstream filename_vtk;
  std::ostringstream filename_ucd;

  std::cout << "------------------" << std::endl;
  std::cout << "Write solution" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  filename_vtk << filename_basis
	   << Utilities::int_to_string (refinement_cycle, 5)
	   << ".vtk";

  filename_ucd << filename_basis
	   << Utilities::int_to_string (refinement_cycle, 5)
	   << ".ucd";
  
  std::ofstream output_vtk (filename_vtk.str().c_str());
  data_out.write_vtk (output_vtk);

}

// With help of this function, we extract 
// point values for a certain component from our
// discrete solution. We use it to gain the 
// displacements of the structure in the x- and y-directions.
template <int dim>
double StepFlowNSE<dim>::compute_point_value (Point<dim> p, 
					       const unsigned int component) const  
{
 
  Vector<double> tmp_vector(dim+1);
  VectorTools::point_value (dof_handler, 
			    solution, 
			    p, 
			    tmp_vector);
  
  return tmp_vector(component);
}

// Now, we arrive at the function that is responsible 
// to compute the line integrals for the drag and the lift. Note, that 
// by a proper transformation via the Gauss theorem, the both 
// quantities could also be achieved by domain integral computation. 
// Nevertheless, we choose the line integration because deal.II provides
// all routines for face value evaluation. 
template <int dim>
void StepFlowNSE<dim>::compute_drag_lift_tensor()
{
    
  const QGauss<dim-1> face_quadrature_formula (3);
  FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
				    update_values | update_gradients | update_normal_vectors | 
				    update_JxW_values);
  
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_face_q_points = face_quadrature_formula.size();

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<Vector<double> >  face_solution_values (n_face_q_points, 
						      Vector<double> (dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    face_solution_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+1));
  
  Tensor<1,dim> drag_lift_value;
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

   for (; cell!=endc; ++cell)
     {

       // First, we are going to compute the forces that
       // act on the cylinder. We notice that only the fluid 
       // equations are defined here.
       for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	 if (cell->face(face)->at_boundary() && 
	     cell->face(face)->boundary_id()==80)
	   {
	     fe_face_values.reinit (cell, face);
	     fe_face_values.get_function_values (solution, face_solution_values);
	     fe_face_values.get_function_gradients (solution, face_solution_grads);
	 	      
	     for (unsigned int q=0; q<n_face_q_points; ++q)
	       {

		 Tensor<2, 2> pI;
		 pI.clear();
		 pI[0][0] = face_solution_values[q](dim);
		 pI[1][1] = face_solution_values[q](dim);
		 
		 
		 Tensor<2,dim> grad_v;
		 grad_v[0][0] = face_solution_grads[q][0][0];
		 grad_v[0][1] = face_solution_grads[q][0][1];
		 grad_v[1][0] = face_solution_grads[q][1][0];
		 grad_v[1][1] = face_solution_grads[q][1][1];

		 Tensor<2,dim> sigma_fluid;
		 sigma_fluid.clear();
		 sigma_fluid = -pI + density_fluid * kinematic_viscosity * (grad_v + transpose(grad_v));
		 
		 drag_lift_value -= sigma_fluid * 
		   fe_face_values.normal_vector(q) * fe_face_values.JxW(q); 
		 
	       }
	   } // end boundary 80 for fluid
       

     } // end cell

   // 2D-1: 500; 2D-2 and 2D-3: 20 (see Schaefer/Turek 1996)
   if (test_case == "2D-1_NSE")
     {
       drag_lift_value *= 500.0;
     }
   
   std::cout << "Face drag:   "  << "   " << std::setprecision(16) << drag_lift_value[0] << std::endl;
   std::cout << "Face lift:   "  << "   " << std::setprecision(16) << drag_lift_value[1] << std::endl;

   // Reference values for 2D-1
   double reference_value_drag = 5.5787294556197073e+00; // global ref 4
   double reference_value_lift = 1.0610686398307201e-02; // global ref 4

}



// Here, we compute the four quantities of interest
template<int dim>
void StepFlowNSE<dim>::compute_functional_values()
{

  double p_front, p_back, p_diff;
  
  p_front = compute_point_value(Point<dim>(0.15,0.2), dim); // pressure
  p_back  = compute_point_value(Point<dim>(0.25,0.2), dim); // pressure
  
  p_diff = p_front - p_back;





  std::cout << "------------------" << std::endl;
  std::cout << "P-Diff:  "  << "   " << std::setprecision(16) << p_diff << std::endl;
  std::cout << "P-front: "  << "   " << std::setprecision(16) << p_front << std::endl;
  std::cout << "P-back:  "  << "   " << std::setprecision(16) << p_back << std::endl;

  std::cout << "------------------" << std::endl;
  // Compute drag and lift via line integral
  compute_drag_lift_tensor();

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;
}





// As usual, we have to call the run method. It handles
// the output stream to the terminal.
// Second, we define some output skip that is necessary 
// (and really useful) to avoid to much printing 
// of solutions. For large time dependent problems it is 
// sufficient to print only each tenth solution. 
// Third, we perform the time stepping scheme of 
// the solution process.
template <int dim>
void StepFlowNSE<dim>::run () 
{  
  // Defining test cases
  // INFO: only 2D-1_NSE is implemented !!!
  // And also in 2D-1, the 2D-2, and 2D-3 cases from Schaefer/Turek 1996 can be computed!!!
  // 
  //test_case = "Channel_NSE";
  test_case = "2D-1_NSE";
  //test_case = "Cavity_IV";
  //test_case = "L_shaped_IV";
  //test_case = "Venturi_IV";

   // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  set_runtime_parameters ();

  setup_system();

  std::cout << "\n==============================" 
	    << "====================================="  << std::endl;
  std::cout << "Parameters\n" 
	    << "==========\n"
	    << "Density flow: "   <<  density_fluid << "\n"  
	    << "Viscosity:    "   <<  kinematic_viscosity << "\n"
	    << std::endl;
  // More output can be printed here if wished


  // Apply initial conditions
   {
      AffineConstraints<double> constraints;
      //ConstraintMatrix constraints;
      constraints.close();

      std::vector<bool> component_mask (dim+1, true);
      VectorTools::project (dof_handler,
			    constraints,
			    QGauss<dim>(degree+2),
			    InitialValues<dim>(),
			    solution
			    );

      
      output_results (timestep_number,solution);
    }


 
   // Increment time after having prescribed the initial solution
   time += timestep;
   ++timestep_number; 

   // Define a flag that not each *.vtk is printed
  const unsigned int output_skip = 1;

  // Time loop
  do
    { 
      std::cout << "Timestep " << timestep_number 
		<< " (" << time_stepping_scheme 
		<< ")" <<    ": " << time
		<< " (" << timestep << ")"
		<< "\n==============================" 
		<< "=====================================================" 
		<< std::endl; 
      
      std::cout << std::endl;	
      
      // Compute next time step solution by solving
      // the nonlinear system
      old_timestep_solution = solution;

      // Solve nonlinear problem
      newton_iteration ();   

      // Compute functional values
      std::cout << std::endl;
      compute_functional_values();


      // Write solutions 
      if ((timestep_number % output_skip == 0))
	output_results (timestep_number,solution);

      
      // Update time
      time += timestep;      
      ++timestep_number;

    }
  while (timestep_number <= max_no_timesteps);
  
  
}

// The main function looks almost the same
// as in all other deal.II tuturial steps. 
int main () 
{
  try
    {
      deallog.depth_console (0);

      StepFlowNSE<2> problem(1);
      problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      
      return 1;
    }
  catch (...) 
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}




