"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xcomplex<f64>>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xcomplex<f64>>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>):
      %5 = "stablehlo.real"(%arg0) : (tensor<complex<f64>>) -> tensor<f64>
      %6 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %7 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %8 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %9 = "stablehlo.select"(%7, %8, %5) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %10 = "stablehlo.compare"(%5, %5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %11 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
      %12 = "stablehlo.select"(%10, %11, %9) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %13 = "stablehlo.imag"(%arg0) : (tensor<complex<f64>>) -> tensor<f64>
      %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %15 = "stablehlo.compare"(%13, %14) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %17 = "stablehlo.select"(%15, %16, %13) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %18 = "stablehlo.compare"(%13, %13) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %19 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
      %20 = "stablehlo.select"(%18, %19, %17) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %21 = "stablehlo.real"(%arg1) : (tensor<complex<f64>>) -> tensor<f64>
      %22 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %23 = "stablehlo.compare"(%21, %22) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %24 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %25 = "stablehlo.select"(%23, %24, %21) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %26 = "stablehlo.compare"(%21, %21) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %27 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
      %28 = "stablehlo.select"(%26, %27, %25) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %29 = "stablehlo.imag"(%arg1) : (tensor<complex<f64>>) -> tensor<f64>
      %30 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %31 = "stablehlo.compare"(%29, %30) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %32 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %33 = "stablehlo.select"(%31, %32, %29) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %34 = "stablehlo.compare"(%29, %29) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %35 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
      %36 = "stablehlo.select"(%34, %35, %33) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %37 = "stablehlo.compare"(%20, %36) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %38 = "stablehlo.compare"(%12, %28) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %39 = "stablehlo.compare"(%12, %28) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %40 = "stablehlo.and"(%39, %37) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %41 = "stablehlo.or"(%38, %40) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%41) : (tensor<i1>) -> ()
    }) : (tensor<5x7xcomplex<f64>>) -> tensor<5x7xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xcomplex<f64>>, tensor<5x7xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<5x7xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(0.64944264624640258,1.8644958567225149), (-1.1994910003008084,-2.3947153534106054), (0.87678242649798709,3.4186975878474462), (-4.941559697600062,3.3040654072929283), (1.7344473357004015,1.5114585492183048), (3.9110869865337605,-6.5450301921577632), (-0.24921953991846346,1.0605301430490701)], [(-0.39456495128517594,2.0464516580769061), (-0.60532504429574174,0.14739522385347475), (4.7120907212489795,-3.7016351244260384), (0.30677575551742364,-3.5517602003857069), (4.7934713839026593,2.2411522548710421), (0.73813718680452056,-7.5334013751630948), (-0.74477976131012891,-3.2448148397921441)], [(4.4030828729636671,1.8617995584501481), (1.4776385552561986,-0.12614478725582773), (0.31361908711910358,0.85138624510558436), (0.55218300919771535,-0.88608320277690233), (4.9668445386980427,1.132838136625913), (-3.6894334284890551,-2.3681588926945638), (4.1082916491323447,1.5775315939191725)], [(0.69313580871902969,-2.1876246473781782), (-5.3367303481182207,-2.3690572089334503), (-6.5735594674279181,-5.4145718413220916), (2.3084125663360737,-0.42271919316078965), (-2.9888835466028341,-1.1330456020838757), (-0.20012046711908324,-1.4094380287854835), (-1.188100880018089,-3.5018452039490056)], [(3.1605309513611379,-0.18253662964002815), (-0.94389012989099152,4.1663781255558483), (1.0883816404967923,1.920608285851281), (-5.2918227675522331,0.8041070865420803), (1.5068728813231811,-0.68677102184695249), (0.49901319800208671,-1.5155025860734601), (1.4989977944138713,2.6472286983937945)]]> : tensor<5x7xcomplex<f64>>}> : () -> tensor<5x7xcomplex<f64>>
    "func.return"(%1) : (tensor<5x7xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-0.39456495128517594,2.0464516580769061), (-5.3367303481182207,-2.3690572089334503), (-6.5735594674279181,-5.4145718413220916), (-5.2918227675522331,0.8041070865420803), (-2.9888835466028341,-1.1330456020838757), (-3.6894334284890551,-2.3681588926945638), (-1.188100880018089,-3.5018452039490056)], [(0.64944264624640258,1.8644958567225149), (-1.1994910003008084,-2.3947153534106054), (0.31361908711910358,0.85138624510558436), (-4.941559697600062,3.3040654072929283), (1.5068728813231811,-0.68677102184695249), (-0.20012046711908324,-1.4094380287854835), (-0.74477976131012891,-3.2448148397921441)], [(0.69313580871902969,-2.1876246473781782), (-0.94389012989099152,4.1663781255558483), (0.87678242649798709,3.4186975878474462), (0.30677575551742364,-3.5517602003857069), (1.7344473357004015,1.5114585492183048), (0.49901319800208671,-1.5155025860734601), (-0.24921953991846346,1.0605301430490701)], [(3.1605309513611379,-0.18253662964002815), (-0.60532504429574174,0.14739522385347475), (1.0883816404967923,1.920608285851281), (0.55218300919771535,-0.88608320277690233), (4.7934713839026593,2.2411522548710421), (0.73813718680452056,-7.5334013751630948), (1.4989977944138713,2.6472286983937945)], [(4.4030828729636671,1.8617995584501481), (1.4776385552561986,-0.12614478725582773), (4.7120907212489795,-3.7016351244260384), (2.3084125663360737,-0.42271919316078965), (4.9668445386980427,1.132838136625913), (3.9110869865337605,-6.5450301921577632), (4.1082916491323447,1.5775315939191725)]]> : tensor<5x7xcomplex<f64>>}> : () -> tensor<5x7xcomplex<f64>>
    "func.return"(%0) : (tensor<5x7xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

