"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xf32>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xf32>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xf32>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.79307747, 4.01534319, 1.2721169], [2.369820e+00, -6.8831439, 1.89294529], [-2.88248158, -0.482540667, -2.66761327], [1.44799101, -0.830354094, 1.0569191]]> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[(1.9936646086611565,-4.4686441038919975), (2.5060465240138097,-3.4576058778526302), (-4.0790969821696059,-2.7514021321086082), (4.6903539903535112,0.74026828593364991), (0.54453231472439145,0.7038216545151238), (-5.5889174788154463,-0.26051787400778836)], [(-1.5754633288058864,-4.80014673987049), (-3.3485801824303252,1.2929738478470481), (3.0242442481728569,2.8853021012792994), (-0.81355011609504623,3.8186922544880586), (1.4963293946323657,-3.3909877855979178), (0.69510131773620465,0.20959187408916286)], [(1.2739432598493683,1.2589379364646724), (0.046603377154599758,1.062839056001927), (3.8422643529046963,-3.9406242510143805), (3.1864287910149205,2.3616371328874202), (0.063084595015577005,-4.5190950214908208), (1.3268804624981352,-0.57396781493133597)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xf32>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-10.273880896310224,-5.1914511319141843), (-20.385995767660425,16.20115033206568), (28.424421834379043,14.257422740012466), (-12.313695055185143,16.270011772708479), (4.5676060836833043,-21.330625262266604), (20.089296887415649,0.83907575113576394)], [(17.980272195234658,24.833318699785039), (29.075856374859672,-15.081732840632089), (-23.209838237582304,-33.839663802673805), (22.746813137712639,-20.05985574126781), (-8.8895912264953996,16.454187979681052), (-15.517499353824345,-3.1465212005812782)], [(-8.3848643181379288,11.838690761810394), (-5.7321266038957788,6.5073292061051742), (0.048925690352806583,17.050651886448765), (-21.627447661899517,-10.27641855218822), (-2.4599294510783216,11.662734395033473), (12.234933076741143,2.1809255275136592)], [(5.541455824121222,-1.1541394572870547), (6.4584961108540728,-4.9568734718136769), (-4.3567267939739605,-10.544749004120625), (10.834922541600111,0.39709446452046349), (-0.38733002801043215,-0.94146981038795418), (-7.2674772085013899,-1.1579005564324787)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

