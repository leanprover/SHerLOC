"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %14 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xcomplex<f32>>
    %15 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xcomplex<f32>>
    %16 = "func.call"(%14) <{callee = @cummin}> : (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>
    "stablehlo.custom_call"(%16, %15) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<8x9xcomplex<f32>>, tensor<8x9xcomplex<f32>>) -> ()
    "func.return"(%16) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %13 = "stablehlo.constant"() <{value = dense<[[(-3.35083818,-1.8786428), (-3.12827182,0.761444747), (-2.26549435,2.24021697), (3.56004572,-4.97253036), (-2.4316442,-4.20448923), (0.170109183,6.39999866), (6.173610e-01,-2.94766331), (2.09143424,-0.430239886), (0.184019849,-1.04114234)], [(1.0007664,3.96333742), (-0.375178367,4.338460e+00), (2.38607645,-6.70219755), (6.30203676,7.06070948), (2.74725294,-4.6113553), (1.89621162,-0.0594967902), (0.698482156,-0.755232274), (-0.53656733,-3.02584958), (-3.17357135,2.29625154)], [(-1.66062677,-2.184590e+00), (6.85014582,0.750131249), (3.77957392,1.5007211), (2.16925883,2.1192627), (3.85788727,1.68599927), (1.48397613,-0.921808421), (-1.38941407,1.14481723), (4.75893116,-5.54427767), (6.82984638,-2.48020554)], [(1.80429149,-2.60177183), (5.93824434,-3.67923903), (-5.18395758,-0.989263772), (-1.01718009,-2.77182722), (4.48587179,0.303576589), (-5.15712309,-4.38469267), (-1.1919378,-0.804852366), (4.89017248,-1.88603735), (-7.51206636,-0.0206586514)], [(-0.873850643,1.75343919), (0.82626754,4.88902664), (-3.41702795,5.93336582), (-4.27905083,2.61049724), (0.534165502,-5.369380e-01), (2.8932023,-1.53613877), (-2.66897416,-1.50824368), (-2.49225426,1.73183668), (6.57836199,4.91882181)], [(0.147815481,2.76275754), (0.845789313,4.46670246), (-2.41715074,5.06512833), (1.16373086,-0.432805598), (-1.70239091,2.69099784), (-3.78774166,1.68323612), (-1.24278891,5.50744629), (0.230480835,0.460102499), (-4.23661947,2.71160746)], [(-6.69632769,2.92194128), (0.155029729,2.31064415), (1.80711317,-3.17149568), (2.16257381,-0.156550273), (-2.67860436,-3.01249933), (2.05095363,1.06082428), (1.73940551,1.66374731), (-2.80290508,1.16762614), (-2.32429743,0.0290495921)], [(-2.7389636,1.22551143), (-1.42216611,-3.11486387), (3.35800767,-3.20950294), (2.87812352,-3.75614452), (0.0551478416,0.911040306), (2.56124187,-0.827129244), (-3.09243584,-1.39052391), (1.93282235,-6.80695247), (-0.891473055,-0.418019027)]]> : tensor<8x9xcomplex<f32>>}> : () -> tensor<8x9xcomplex<f32>>
    "func.return"(%13) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[[(-3.35083818,-1.8786428), (-3.12827182,0.761444747), (-2.26549435,2.24021697), (3.56004572,-4.97253036), (-2.4316442,-4.20448923), (0.170109183,6.39999866), (6.173610e-01,-2.94766331), (2.09143424,-0.430239886), (0.184019849,-1.04114234)], [(-3.35083818,-1.8786428), (-3.12827182,0.761444747), (-2.26549435,2.24021697), (3.56004572,-4.97253036), (-2.4316442,-4.20448923), (0.170109183,6.39999866), (6.173610e-01,-2.94766331), (-0.53656733,-3.02584958), (-3.17357135,2.29625154)], [(-3.35083818,-1.8786428), (-3.12827182,0.761444747), (-2.26549435,2.24021697), (2.16925883,2.1192627), (-2.4316442,-4.20448923), (0.170109183,6.39999866), (-1.38941407,1.14481723), (-0.53656733,-3.02584958), (-3.17357135,2.29625154)], [(-3.35083818,-1.8786428), (-3.12827182,0.761444747), (-5.18395758,-0.989263772), (-1.01718009,-2.77182722), (-2.4316442,-4.20448923), (-5.15712309,-4.38469267), (-1.38941407,1.14481723), (-0.53656733,-3.02584958), (-7.51206636,-0.0206586514)], [(-3.35083818,-1.8786428), (-3.12827182,0.761444747), (-5.18395758,-0.989263772), (-4.27905083,2.61049724), (-2.4316442,-4.20448923), (-5.15712309,-4.38469267), (-2.66897416,-1.50824368), (-2.49225426,1.73183668), (-7.51206636,-0.0206586514)], [(-3.35083818,-1.8786428), (-3.12827182,0.761444747), (-5.18395758,-0.989263772), (-4.27905083,2.61049724), (-2.4316442,-4.20448923), (-5.15712309,-4.38469267), (-2.66897416,-1.50824368), (-2.49225426,1.73183668), (-7.51206636,-0.0206586514)], [(-6.69632769,2.92194128), (-3.12827182,0.761444747), (-5.18395758,-0.989263772), (-4.27905083,2.61049724), (-2.67860436,-3.01249933), (-5.15712309,-4.38469267), (-2.66897416,-1.50824368), (-2.80290508,1.16762614), (-7.51206636,-0.0206586514)], [(-6.69632769,2.92194128), (-3.12827182,0.761444747), (-5.18395758,-0.989263772), (-4.27905083,2.61049724), (-2.67860436,-3.01249933), (-5.15712309,-4.38469267), (-3.09243584,-1.39052391), (-2.80290508,1.16762614), (-7.51206636,-0.0206586514)]]> : tensor<8x9xcomplex<f32>>}> : () -> tensor<8x9xcomplex<f32>>
    "func.return"(%12) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>, sym_name = "cummin", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xcomplex<f32>>):
    %0 = "stablehlo.constant"() <{value = dense<(0x7F800000,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<complex<f32>>) -> tensor<complex<f32>>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<complex<f32>>, %arg2: tensor<complex<f32>>):
      %3 = "stablehlo.real"(%arg1) : (tensor<complex<f32>>) -> tensor<f32>
      %4 = "stablehlo.real"(%arg2) : (tensor<complex<f32>>) -> tensor<f32>
      %5 = "stablehlo.compare"(%3, %4) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = "stablehlo.compare"(%3, %4) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = "stablehlo.imag"(%arg1) : (tensor<complex<f32>>) -> tensor<f32>
      %8 = "stablehlo.imag"(%arg2) : (tensor<complex<f32>>) -> tensor<f32>
      %9 = "stablehlo.compare"(%7, %8) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = "stablehlo.select"(%5, %9, %6) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%10, %arg1, %arg2) : (tensor<i1>, tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
      "stablehlo.return"(%11) : (tensor<complex<f32>>) -> ()
    }) : (tensor<8x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    "func.return"(%2) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

