"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 0, 0], [2, 0, -2], [2, -2, -2], [2, 0, 0]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[(-4.2350784816667506,4.3798841507514377), (1.8657300604800531,0.46782141538971833), (-0.80250143597150658,1.1732318473225631), (-3.7655237883695927,-5.8164533740084572), (-1.1592618183468728,1.6223864052200678), (-0.61511846883459076,0.25475632869192694)], [(4.8285962951303887,1.645810477955151), (-4.3338864616355393,-4.3688050934421687), (-2.9970548129215291,0.97156177656526976), (2.4843316557750699,1.5441591592779584), (-5.2721090146291072,-2.0934023339360417), (0.49214774708535214,7.2179773411323307)], [(5.1004885552852901,0.84855471270999572), (4.0032172549997629,-3.4333931615997253), (3.6049624757257943,-2.5226047670163707), (0.32341545203008487,-1.3692249727636305), (-2.9497222560306797,-0.81114762110035254), (-2.7278259871726207,0.68396660488524508)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(-18.671134073904081,7.0626588760828835), (-4.2749743890394196,7.8024291539788874), (-8.8149278233946013,7.3916732286778677), (-8.1778784807993556,-8.8944568024896533), (3.5809208753676138,4.8670680526408407), (4.2254150366760594,-0.85842055238663628)], [(-28.328326664164859,3.7710379201725819), (4.3927985342316589,16.540039340863224), (-2.8208181975515441,5.4485496755473282), (-13.146541792349495,-11.98277512104557), (14.125138904625828,9.0538727205129241), (3.2411195425053556,-15.294375234651298)], [(-8.4701569633335012,8.7597683015028753), (3.7314601209601062,0.93564283077943666), (-1.6050028719430132,2.3464636946451263), (-7.5310475767391853,-11.632906748016914), (-2.3185236366937456,3.2447728104401357), (-1.2302369376691815,0.50951265738385387)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

