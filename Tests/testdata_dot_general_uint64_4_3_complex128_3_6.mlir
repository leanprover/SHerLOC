"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui64>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui64>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui64>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[5, 0, 0], [4, 3, 1], [0, 1, 1], [3, 2, 1]]> : tensor<4x3xui64>}> : () -> tensor<4x3xui64>
    %2 = "stablehlo.constant"() <{value = dense<[[(-3.9018996874182315,3.159414026193232), (1.3555319279595275,-0.052908537308914885), (0.99249887111152157,-1.7946815300193639), (-2.8521186266692755,5.8948912378065614), (-1.5567094366342697,7.5459875004100923), (0.67709275925978751,1.5408649085206472)], [(-1.362182852233,-2.6802952292433839), (-2.804684432985864,-0.0030541114780836404), (3.4368770468153862,3.1902895701285408), (1.8950410804889843,1.5086980494423943), (-2.5668018581266621,5.4700039304387573), (5.2451468933923096,1.4696787021717228)], [(1.5912479307192151,1.674966552899114), (-1.2428421803666405,-2.5571140230215952), (-6.8898488758491805,0.12482905952402451), (-6.5057844487657182,-3.5999036534105167), (0.2054169978876626,-0.26427713505750355), (-2.5886828869929244,2.8370394138760107)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xui64>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-19.509498437091157,15.79707013096616), (6.777659639797637,-0.26454268654457441), (4.9624943555576078,-8.973407650096819), (-14.260593133346378,29.474456189032807), (-7.7835471831713487,37.729937502050461), (3.3854637962989376,7.7043245426032358)], [(-18.102899375652708,6.271736969941891), (-4.2347677674861224,-2.7779105066915055), (7.3907777490430639,2.5169716498321901), (-12.229135713975868,24.50575544614291), (-13.721826323029402,46.329684657899136), (15.855128830223155,13.409535154473769)], [(0.22906507848621516,-1.0053286763442699), (-4.0475266133525043,-2.5601681344996789), (-3.4529718290337943,3.3151186296525652), (-4.6107433682767338,-2.0912056039681222), (-2.3613848602389993,5.2057267953812536), (2.6564640063993852,4.3067181160477332)], [(-12.838816836001479,5.7926181729920412), (-2.7856152624597854,-2.7219478579045071), (2.9614018311161576,1.1213636097230144), (-11.272058167795576,17.102166158893954), (-9.5983150282684714,33.313693227050287), (9.9328891775710577,10.398991543781399)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

